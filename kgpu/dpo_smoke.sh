#!/usr/bin/env bash
# dpo_smoke.sh — LoCo(B안) 배선 점검 1단계: 원본 Qwen3.5-9B 에 이미지 포함 DPO 2 step 이 도는가(학습 목적 아님).
#   쌍 = base_rollouts.jsonl 의 같은 문항 응답 중 짧은 것(chosen) · 가장 긴 것(rejected). 정답 판정은 하지 않는다(배선·메모리만 본다).
# 전제(~/files/kbds): setup_venv.sh · rubric/rubric_inputs.tar · rubric/base_rollouts.jsonl
# env(선택): PAIRS(16) MAXLEN(10240) STEPS(2) MODEL_HF(Qwen/Qwen3.5-9B)
# 실행: setsid nohup bash kgpu/run_pod.sh dpo_smoke.sh HF_TOKEN > logs/dpo_smoke.out 2>&1 < /dev/null & disown
# 판정: DPO_EXIT=0 · loss 기록 · 최대 메모리. 로그는 ~/files/kbds/logs/dpo_smoke_*.log
set -uo pipefail
F=/root/files/kbds; W=/root/w; VENV=/root/venv
mkdir -p "$W" "$F/logs"; exec > >(tee "$W/dpo.log") 2>&1
trap 'cp "$W/dpo.log" "$F/logs/dpo_smoke_$(date +%m%d-%H%M).log"' EXIT
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTORCH_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
stamp "1) 입력·쌍 만들기"
mkdir -p "$W/in" && tar xf "$F/rubric/rubric_inputs.tar" -C "$W/in" --no-same-owner || { echo "❌ rubric_inputs.tar"; exit 1; }
stamp "2) venv"; . "$F/setup_venv.sh"
$PY - "$W/in" "$F/rubric/base_rollouts.jsonl" "$W/pairs.jsonl" "${PAIRS:-16}" <<'PYEOF' || { echo "❌ 쌍 만들기"; exit 1; }
import json, sys
src, roll, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
SYS = ("You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, "
       "keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>.")
strip = lambda t: t[len('<think>\n'):] if t.startswith('<think>\n') else t   # 생성 프롬프트가 '<think>\n' 로 끝난다
k = 0
with open(out, 'w') as f:
    for l in open(roll):
        r = json.loads(l)
        c = sorted((x for x, fin in zip(r['completions'], r['finish']) if fin == 'stop' and '<answer>' in x), key=len)
        if len(c) < 2: continue
        f.write(json.dumps({'messages': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': '<image>' + r['question']}, {'role': 'assistant', 'content': strip(c[0])}],
                            'rejected_response': strip(c[-1]), 'images': [f"{src}/images/{r['image']}"]}, ensure_ascii=False) + '\n'); k += 1
        if k >= n: break
print(f'쌍 {k}개')
PYEOF
stamp "3) 모델 ← HF"
# 지난 시도에서 마지막 조각에서 멈췄다 → 8분 제한을 두고 3번까지 재시도(2번째부터 xet 끔)
for TRY in 1 2 3; do
  HF_HUB_DISABLE_XET=$([ $TRY -gt 1 ] && echo 1 || echo 0) timeout 480 $PY -c "from huggingface_hub import snapshot_download as s;s('${MODEL_HF:-Qwen/Qwen3.5-9B}',local_dir='$W/model')" && break
  echo "내려받기 ${TRY}번째 실패 — 다시 시도"; [ $TRY = 3 ] && { echo "❌ model"; exit 1; }
done
stamp "4) swift rlhf dpo ${STEPS:-2} step (LoRA r256, 어텐션 q/k/v/o, beta 0.1 — 논문 설정)"
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 NPROC_PER_NODE=1 "$VENV/bin/swift" rlhf --rlhf_type dpo \
  --model "$W/model" --model_type qwen3_5 --dataset "$W/pairs.jsonl" --enable_thinking true \
  --tuner_type lora --lora_rank 256 --lora_alpha 512 --lora_dropout 0.05 --target_modules q_proj k_proj v_proj o_proj --torch_dtype bfloat16 \
  --beta 0.1 --learning_rate 1e-5 --max_length "${MAXLEN:-10240}" --max_pixels 262144 --gradient_checkpointing true \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --attn_impl sdpa \
  --logging_steps 1 --save_strategy no --report_to none --output_dir "$W/run" --max_steps "${STEPS:-2}"
echo "DPO_EXIT=$?"
stamp "done"
