#!/bin/bash
# =============================================================================
# launch_gspo_ab.sh — GSPO vs dr_grpo A/B (Stage-2 최신기법 검증)
# -----------------------------------------------------------------------------
#  목적: dr_grpo(현 1-epoch 재학습, 58892 체인)와 **완전 동일 조건**에서 RECIPE 만 gspo 로 바꿔,
#        GSPO(시퀀스레벨 IS, 2507.18071)가 우리 데이터에서 dr_grpo 를 이기는지 실증.
#        "최신이라서" 채택하지 않고, dr_grpo 가 자리를 얻은 것과 동일한 clean A/B 로 판정.
#
#  동일하게 유지(clean A/B): init(sft_rft_coldstart_merged) · 데이터(trainonly, 홀드아웃 제외)
#        · 보상(accuracy_mix/format_think/soft_overlong 1.0/0.2/0.2) · LoRA r16/a32 · num_gen 4
#        · max_completion 6144 · 공통코어(dynamic_sample+overlong_filter).
#  변경만: RECIPE=gspo → --importance_sampling_level sequence, 작은 clip(3e-4/4e-4).
#
#  비교창: MAX_STEPS=600 → dr_grpo 판정에 쓴 step 501~600 구간을 동일하게 비교(싼 판정).
#          600 step × ~365s ≈ 61h < 70h → 단일 잡 완결(체인 불필요). save_steps=50.
#
#  판정(완주 후): 두 런의 step501~600 구간평균 Acc/reward/mean_len 비교
#          + 필요시 각 checkpoint 병합해 층화 홀드아웃 벤치마크(40_eval_compare).
#
#  ⚠️ QOS 사용자당 제출 6잡 상한. dr_grpo 체인이 슬롯을 다 쓰면 이 스크립트 제출 실패 →
#     dr_grpo 버퍼 잡 1개(체인 맨 뒤)를 취소해 슬롯 확보(5잡으로도 dr_grpo 1-epoch 완주 충분).
#     GSPO 완료 후 슬롯이 비면 dr_grpo 버퍼 재추가 가능.
#
#  사용:  bash scripts/launch_gspo_ab.sh
#         MAX_STEPS=1000 bash scripts/launch_gspo_ab.sh   # 더 긴 창(2잡 필요할 수 있음)
# =============================================================================
set -uo pipefail   # -e 제외: sbatch 실패 시 버퍼 출력 유실 방지(명시적으로 에러 처리)
cd /home01/k252a01/kbds_project

DATASET="${DATASET_FILE:-/home01/k252a01/kbds_project/work/data/deepvision103k_trainonly.jsonl}"
OUTDIR="${OUTPUT_DIR:-/home01/k252a01/kbds_project/work/checkpoints/grpo_general_adv_gspo_he}"
MAX="${MAX_STEPS:-600}"

[[ -f "$DATASET" ]] || { echo "[gspo-ab] ❌ 데이터 없음: $DATASET"; exit 1; }

echo "[gspo-ab] GSPO A/B 제출 — fresh(init=sft_rft_coldstart_merged), trainonly, MAX_STEPS=$MAX"
echo "[gspo-ab] OUTPUT_DIR=$OUTDIR   (dr_grpo 는 ..._dr_grpo_he/ 별도)"

# QOS 6잡 상한이면 sbatch 가 실패함 → 에러를 그대로 보여주고 버퍼 취소 안내
OUT=$(sbatch --parsable \
      --export="ALL,RECIPE=gspo,DATASET_FILE=$DATASET,OUTPUT_DIR=$OUTDIR,MAX_STEPS=$MAX" \
      scripts/21_rlvr_grpo_adv.slurm 2>&1)
RC=$?
if [[ $RC -ne 0 || -z "$OUT" ]]; then
  echo "[gspo-ab] ❌ 제출 실패(rc=$RC): $OUT"
  echo "[gspo-ab]   QOSMaxSubmitJobPerUserLimit 이면 dr_grpo 버퍼 잡을 1개 취소 후 재실행:"
  echo "[gspo-ab]   squeue -u \$USER  # 맨 뒤 잡 확인 →  scancel <버퍼JID>  →  bash scripts/launch_gspo_ab.sh"
  exit 1
fi
JID="$OUT"
echo "[gspo-ab] GSPO job = $JID"
echo "[gspo-ab] 모니터: squeue -j $JID ; ls $OUTDIR/*/checkpoint-*"
echo "[gspo-ab] 완주 후: dr_grpo step501~600 vs gspo step501~600 구간평균 비교 → 승자 채택"
