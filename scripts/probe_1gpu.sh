#!/bin/bash
# =============================================================================
# probe_1gpu.sh — 배선 검증만. debug-1gpu(A100 40GB · 유휴) 에서 3 step 돌린다.
# -----------------------------------------------------------------------------
#  왜: 8gpu 3노드가 전부 5일짜리 잡에 물려 최단 시작이 4일 뒤다(2026-08-13 실측).
#      스모크에 4일, 본실행에 또 4~5일 = 8~9일. 그런데 스모크가 보려던 4가지 중
#      **3가지는 GPU 개수와 무관**하다. 그 3가지를 지금 유휴 노드에서 본다.
#
#        ✅ 볼 수 있는 것
#           · configs/accuracy.py 가 로드되고 단계형 FormatThink 가 값을 내는가
#           · recipe=stable 인자를 ms-swift 가 받는가 (특히 log_rollout_offpolicy_metrics)
#           · rollout_correction/* 지표가 실제로 로그에 찍히는가
#           · DOMAIN 프리셋 분기 + 도메인 jsonl 로딩 + --lora_dropout
#        ❌ 못 보는 것
#           · **step time** — 8 GPU 기준이 아니고 offload 때문에 몇 배 느리다. 읽지 말 것.
#           · 8 GPU DDP/NCCL 경로. 이건 73924 에서 이미 검증된 부분이라 위험이 낮다.
#
#  왜 이 스크립트가 21_rlvr_grpo_adv.slurm 을 그대로 부르는가:
#      축소본을 따로 만들면 **정작 검증하려던 본 스크립트가 검증되지 않는다.**
#      크기 차이는 전부 환경변수로만 준다.
#
#  A100-40GB 메모리 회계 (9B bf16 = 18GB 사본 하나):
#      롤아웃 중 : vLLM 18 + KV 6 = 24GB   (학습가중치·옵티마는 OFFLOAD=1 로 CPU)
#      학습 중   : 학습 18 + 활성값        (vLLM 은 SLEEP_LEVEL=2 로 가중치까지 해제)
#      → 두 사본이 절대 동시에 GPU 에 있지 않다. 대신 매 step 18GB 재동기 = 느리다.
#      CPU RAM 60GB 중 18GB 를 offload 가 쓴다. 여유 있다.
#
#  사용:
#      bash scripts/probe_1gpu.sh              # pmcvqa 3 step
#      DOMAIN=mmk12 bash scripts/probe_1gpu.sh
#      DRY=1 bash scripts/probe_1gpu.sh
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

DOMAIN="${DOMAIN:-pmcvqa}"      # 가장 작은 예산(3072). 새로 넣은 분기라 검증 가치가 가장 높다.
STEPS="${STEPS:-3}"
NUM_GEN="${NUM_GEN:-8}"
STAMP="${STAMP:-$(date +%m%d-%H%M)}"
OUT="$CKPT_DIR/probe1g_${DOMAIN}_${STAMP}"

# 생성배치 = PDTBS × ACCUM × world 가 num_generations 로 나누어떨어져야 한다.
#   world=1 이므로 ACCUM=NUM_GEN → 8 completion = 1 프롬프트/step.
ACCUM="${ACCUM:-$NUM_GEN}"

#  본실행 대비 줄인 것 — **여기서 나온 수치를 본실행에 그대로 옮기지 말 것.**
#    MAX_PIXELS  1003520 → 262144   멀티모달 활성값이 40GB 에서 가장 큰 위험
#    MAX_LEN       10240 →   4096
#    MAX_COMPLETION 3072 →   1024   (속도용. 길이 관련 수치는 이 실행에서 읽지 않는다)
SUBMIT=(sbatch --parsable
  --partition=debug-1gpu --gres=gpu:1 --cpus-per-task=8 --mem=56G
  --time="${WALLTIME:-02:00:00}" --job-name="p1g-$DOMAIN"
  --export=ALL,RECIPE=stable,DOMAIN="$DOMAIN",NUM_GEN="$NUM_GEN",LORA_DROPOUT=0,MAX_STEPS="$STEPS",INIT_MODEL="$CKPT_DIR/sft_mixed_merged",OUTPUT_DIR="$OUT",NPROC_PER_NODE=1,PDTBS=1,ACCUM="$ACCUM",OFFLOAD=1,SLEEP_LEVEL=2,VLLM_UTIL=0.62,VLLM_MAX_LEN=4096,MAX_LEN=4096,MAX_COMPLETION=1024,MAX_PIXELS=262144,OMP_NUM_THREADS=8
  "$PROJ_DIR/scripts/21_rlvr_grpo_adv.slurm")

echo "[probe] debug-1gpu · A100-40GB · $DOMAIN · $STEPS step · num_gen=$NUM_GEN (accum=$ACCUM → 1 프롬프트/step)"
echo "[probe] out= $OUT"
if [[ "${DRY:-0}" == "1" ]]; then echo "[probe] (DRY) ${SUBMIT[*]}"; exit 0; fi

JID=$("${SUBMIT[@]}")
echo "[probe] 제출: job $JID  ·  로그 logs/grpo_adv_${JID}.log"
cat <<EOF

[probe] 판독:
    tail -f logs/grpo_adv_${JID}.log
    grep -E "rewards/|rollout_correction|completions/|OutOfMemory|Traceback" logs/grpo_adv_${JID}.log

    통과 = 3 step 이 끝까지 돌고 아래가 전부 보인다
      · rewards/AccuracyMix/mean · rewards/FormatThink/mean · rewards/SoftOverlong/mean
      · rollout_correction/* (← recipe=stable 이 실제로 계측을 켰다는 증거)
    ⚠️ step time · reward 값의 크기는 읽지 않는다. 1 프롬프트/step 이라 통계가 아니다.
EOF
