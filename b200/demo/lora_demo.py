#!/usr/bin/env python3
"""lora_demo.py — B200 8장으로 HF 모델을 LoRA 튜닝하는 최소 예제.

torchrun 이 프로세스를 GPU 당 하나씩 띄우고, 각 프로세스가 모델 사본을 하나씩 들고
그래디언트만 서로 합친다(DDP). LoRA 라 학습되는 파라미터는 전체의 0.1% 수준이다.

  torchrun --nproc_per_node=8 lora_demo.py

데이터셋은 일부러 코드 안에 넣었다 — 데모의 목적은 플랫폼 사용법이고,
외부 데이터 다운로드가 실패하면 배우려던 것과 무관한 곳에서 막힌다.
실제 학습에서는 datasets.load_dataset(...) 으로 바꾸면 된다.
"""
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = os.environ.get("DEMO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUT = os.environ.get("DEMO_OUT", "./lora_demo_out")
EPOCHS = int(os.environ.get("DEMO_EPOCHS", "3"))
BS = int(os.environ.get("DEMO_BS", "4"))
MAXLEN = 256

# ── 학습 데이터 (24쌍) ────────────────────────────────────────────────────────
PAIRS = [
    ("대한민국의 수도는?", "서울입니다."),
    ("물의 화학식은?", "H2O 입니다."),
    ("1 더하기 1은?", "2 입니다."),
    ("빛의 속도는?", "초속 약 30만 킬로미터입니다."),
    ("파이썬에서 리스트를 뒤집는 방법은?", "list.reverse() 또는 list[::-1] 을 씁니다."),
    ("지구에서 가장 큰 대양은?", "태평양입니다."),
    ("DNA 의 이중나선을 발견한 사람은?", "왓슨과 크릭입니다."),
    ("섭씨 100도는 화씨 몇 도인가?", "212도입니다."),
    ("행렬 곱셈은 교환법칙이 성립하는가?", "일반적으로 성립하지 않습니다."),
    ("GPU 가 CPU 보다 학습에 유리한 이유는?", "같은 연산을 대량으로 병렬 처리하기 때문입니다."),
    ("LoRA 가 전체 미세조정보다 가벼운 이유는?", "원래 가중치를 고정하고 저차원 행렬만 학습하기 때문입니다."),
    ("DDP 는 무엇을 주고받는가?", "각 프로세스가 계산한 그래디언트를 all-reduce 로 합칩니다."),
] * 2

TPL = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"


class Pairs(Dataset):
    def __init__(self, tok):
        self.rows = []
        for q, a in PAIRS:
            enc = tok(TPL.format(q=q, a=a), truncation=True, max_length=MAXLEN,
                      padding="max_length", return_tensors="pt")
            ids = enc["input_ids"][0]
            labels = ids.clone()
            labels[enc["attention_mask"][0] == 0] = -100   # 패딩은 손실에서 뺀다
            self.rows.append({"input_ids": ids, "attention_mask": enc["attention_mask"][0],
                              "labels": labels})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")

    if rank == 0:
        print(f"[demo] world_size={world} · model={MODEL}", flush=True)
    print(f"[demo] rank {rank} → {torch.cuda.get_device_name(dev)} "
          f"({torch.cuda.get_device_properties(dev).total_memory/2**30:.0f} GiB)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # attn_implementation='sdpa': 이 노드에 flash-attn 이 없다(CLAUDE.md §1.4 함정⑤).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    if rank == 0:
        model.print_trainable_parameters()

    ddp = DDP(model, device_ids=[dev.index])
    ds = Pairs(tok)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=BS, sampler=sampler)
    opt = torch.optim.AdamW([p for p in ddp.parameters() if p.requires_grad], lr=2e-4)

    ddp.train()
    for ep in range(EPOCHS):
        sampler.set_epoch(ep)
        tot = n = 0
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss = ddp(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item(); n += 1
        # 각 rank 의 평균 손실을 다시 평균한다 — 전체 손실을 보려면 이 단계가 필요하다.
        t = torch.tensor([tot, n], dtype=torch.float64, device=dev)
        dist.all_reduce(t)
        if rank == 0:
            print(f"[demo] epoch {ep+1}/{EPOCHS}  loss {t[0].item()/t[1].item():.4f}", flush=True)

    if rank == 0:
        ddp.module.save_pretrained(OUT)          # 어댑터만 저장된다(수 MB)
        sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
        print(f"[demo] 어댑터 저장 → {OUT}  ({sz/2**20:.1f} MiB)", flush=True)
        print("[demo] DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
