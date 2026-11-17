#!/bin/bash
# 39_loco_dpo.slurm — 풀이 줄이기(LoCo) DPO 를 KISTI 8gpu 에서 돌린다. kgpu/loco_dpo.sh 와 같은 설정을 8 GPU 로 나눴다.
#   유효 배치 8 = 8 GPU × 배치 1 × 누적 1 (kgpu 는 1 GPU × 누적 8) · LoRA r256 q/k/v/o · beta 0.1 · lr 1e-5 cosine · 1 epoch
# 왜(2026-11-13): 6차 짝의 좋은 답이 형식을 어겨 다시 학습한다(짝 = work/loco/v6f, scripts/fmt_strict.py 로 고침). kgpu rtxpro6000 은 자리가 없었다.
# 전제: $PAIRS(pairs.jsonl, 이미지 경로는 절대경로) · HF 캐시의 Qwen/Qwen3.5-9B(c202236)
# 사용: sbatch --export=ALL,PAIRS=$WORK_DIR/loco/v6f/pairs.jsonl,OUT_NAME=loco_adapter_v6f scripts/39_loco_dpo.slurm
#SBATCH --job-name=loco-dpo
#SBATCH --partition=8gpu
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=/home01/k266a01/kbds_project/logs/loco_dpo_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/loco_dpo_%j.log
set -uo pipefail
source /home01/k266a01/kbds_project/scripts/00_common.sh
: "${PAIRS:?PAIRS 필요}" "${OUT_NAME:?OUT_NAME 필요}"
OUT=$CKPT_DIR/$OUT_NAME
[ -s "$PAIRS" ] || { echo "❌ 짝 파일 없음: $PAIRS"; exit 1; }
[ -e "$OUT" ] && { echo "❌ 출력이 이미 있다(덮지 않는다): $OUT"; exit 1; }
echo "[loco-dpo] PAIRS=$PAIRS ($(wc -l < "$PAIRS") 쌍) OUT=$OUT"
nvidia-smi -L
export NPROC_PER_NODE=8 PYTORCH_ALLOC_CONF=expandable_segments:True
run_py swift rlhf --rlhf_type dpo \
  --model "$BASE_MODEL" --model_type "$MODEL_TYPE" --dataset "$PAIRS" --enable_thinking true \
  --tuner_type lora --lora_rank 256 --lora_alpha 512 --lora_dropout 0.05 --target_modules ${TARGETS:-q_proj k_proj v_proj o_proj} --torch_dtype bfloat16 \
  --beta 0.1 --learning_rate "${LR:-1e-5}" --lr_scheduler_type cosine --warmup_ratio 0.1 --num_train_epochs "${EPOCHS:-1}" \
  --max_length "${MAXLEN:-10240}" --max_pixels 262144 --gradient_checkpointing true \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --attn_impl sdpa \
  --logging_steps 1 --save_strategy steps --save_steps "${SAVE_STEPS:-25}" --save_total_limit "${SAVE_LIMIT:-4}" \
  --report_to none --seed 42 --data_seed 42 --output_dir "$OUT"
echo "DPO_EXIT=$?"
ls -d "$OUT"/*/checkpoint-* 2>/dev/null
