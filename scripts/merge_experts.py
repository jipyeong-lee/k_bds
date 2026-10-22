#!/usr/bin/env python3
"""merge_experts.py — Stage-2 전문가 LoRA 여러 개를 가중치 산술로 base 에 합친다(계단 0, 학습 없음).

  W = W_base + Σ wᵢ·ΔWᵢ,   ΔWᵢ = Bᵢ·Aᵢ·(αᵢ/rᵢ)            (--method linear)
  TIES(--method ties): τᵢ = wᵢ·ΔWᵢ 를 텐서별 상위 density 만 남기고, 부호 다수결(sign Σ τᵢ)과 같은 부호인
  값만 평균한다(Yadav et al. 2023). 근거·판정 기준 = docs/deepseek_v4_pipeline_adoption.md §8.

base 의 safetensors shard 를 하나씩 읽고 LoRA 가 걸린 텐서만 고쳐 같은 이름으로 쓴다 → 모델 클래스를
불러오지 않으므로 Qwen3.5 VL 의 비전 타워·processor 파일이 그대로 보존된다. ΔW 는 모듈 단위로 만들고 버린다
(전체 ΔW 를 한 번에 들면 어댑터당 ~36GB).

예) python scripts/merge_experts.py --base work/checkpoints/sft_mixed_merged \
      --adapters dv/checkpoint-2700,mm/checkpoint-350,pmc/checkpoint-4896 --weights 0.5,0.5,0.5 \
      --method linear --out work/checkpoints/merged_lin05
"""
import argparse, json, os, shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_adapter(path):
    cfg = json.load(open(os.path.join(path, 'adapter_config.json')))
    assert not cfg.get('use_rslora') and not cfg.get('use_dora'), f'{path}: rslora/dora 미지원'
    scale = cfg['lora_alpha'] / cfg['r']
    mods = {}
    with safe_open(os.path.join(path, 'adapter_model.safetensors'), 'pt') as f:
        for k in f.keys():
            # base_model.model.<X>.lora_A.weight → base 키 <X>.weight
            stem, part = k.removeprefix('base_model.model.').split('.lora_')
            mods.setdefault(stem + '.weight', {})[part[0]] = f.get_tensor(k)
    assert all(set(v) == {'A', 'B'} for v in mods.values()), f'{path}: lora_A/B 짝이 안 맞음'
    return mods, scale


def ties(taus, density):
    trimmed = []
    for t in taus:
        k = max(1, int(t.numel() * density))
        thr = t.abs().flatten().kthvalue(t.numel() - k + 1).values
        trimmed.append(torch.where(t.abs() >= thr, t, torch.zeros_like(t)))
    st = torch.stack(trimmed)
    sign = torch.sign(st.sum(0))
    agree = (torch.sign(st) == sign) & (st != 0)
    return (st * agree).sum(0) / agree.sum(0).clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--adapters', required=True, help='쉼표 구분 체크포인트 디렉터리')
    ap.add_argument('--weights', required=True, help='쉼표 구분, 어댑터 수와 같게')
    ap.add_argument('--method', choices=['linear', 'ties'], default='linear')
    ap.add_argument('--density', type=float, default=0.2, help='ties 에서 텐서별로 남길 비율')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    paths, ws = a.adapters.split(','), [float(x) for x in a.weights.split(',')]
    assert len(paths) == len(ws), '--adapters 와 --weights 개수가 다르다'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    adapters = [load_adapter(p) for p in paths]
    targets = set(adapters[0][0])
    assert all(set(m) == targets for m, _ in adapters), '어댑터마다 대상 모듈이 다르다'

    idx = json.load(open(os.path.join(a.base, 'model.safetensors.index.json')))['weight_map']
    missing = targets - set(idx)
    assert not missing, f'base 에 없는 키 {len(missing)}개, 예: {sorted(missing)[:3]}'

    os.makedirs(a.out, exist_ok=True)
    for name in os.listdir(a.base):          # config·tokenizer·processor·index 는 그대로
        if not name.endswith('.safetensors'):
            shutil.copy2(os.path.join(a.base, name), a.out)

    done = 0
    for shard in sorted(set(idx.values())):
        out = {}
        with safe_open(os.path.join(a.base, shard), 'pt') as f:
            meta = f.metadata()
            for k in f.keys():
                w = f.get_tensor(k)
                if k in targets:
                    taus = [wi * s * (m[k]['B'].to(dev) @ m[k]['A'].to(dev)) for (m, s), wi in zip(adapters, ws)]
                    delta = sum(taus) if a.method == 'linear' else ties(taus, a.density)
                    assert delta.shape == w.shape, f'{k}: ΔW {tuple(delta.shape)} ≠ W {tuple(w.shape)}'
                    w = (w.to(dev, torch.float32) + delta).to(w.dtype).cpu()
                    done += 1
                out[k] = w
        save_file(out, os.path.join(a.out, shard), metadata=meta or {'format': 'pt'})
        print(f'[merge] {shard}: {len(out)} tensors', flush=True)

    assert done == len(targets), f'고친 텐서 {done} ≠ LoRA 대상 {len(targets)}'
    with open(os.path.join(a.out, 'merge_config.json'), 'w') as f:
        json.dump({'adapters': paths, 'weights': ws, 'method': a.method,
                   'density': a.density if a.method == 'ties' else None, 'modules': done}, f, indent=1)
    print(f'[merge] done: {done} modules × {len(paths)} adapters, {a.method} → {a.out}')


if __name__ == '__main__':
    main()
