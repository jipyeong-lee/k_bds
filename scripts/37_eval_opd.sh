#!/usr/bin/env bash
# 37_eval_opd.sh — OPD 학생을 ta_w07 기준선과 같은 조건으로 평가한다(KISTI 로그인 노드에서 실행).
#   병합 → holdout(1,772) 평가 + 글로벌 벤치(3,712) 평가 → 짝지은 검정(eval_paired.py).
#   기준선 = ta_w07 의 기존 측정 job 83037(holdout) · 83038(글로벌), 문항별 결과가 logs/ 에 있다.
#   조건: greedy, holdout 최대 출력 8192 · 글로벌 4096(docs/2026-1차/00_overview §1.6 의 조건).
# 사용:  OPD_CKPT=<checkpoint-N 디렉터리> TAG=opd100 bash scripts/37_eval_opd.sh
#        DRY_RUN=1 이면 제출하지 않고 명령만 출력한다.
set -uo pipefail
cd "${PROJ_DIR:-/home01/k266a01/kbds_project}"
: "${OPD_CKPT:?OPD_CKPT 필요}"; TAG=${TAG:?TAG 필요(예 opd100)}
OUT=work/checkpoints/merged_$TAG
[ -f "$OPD_CKPT/adapter_model.safetensors" ] || { echo "❌ 어댑터 없음: $OPD_CKPT"; exit 1; }
for b in logs/eval_items_init_83037.jsonl logs/eval_items_gbench_sft_83038.jsonl; do
  [ -f "$b" ] || { echo "❌ 기준선 문항별 결과 없음: $b"; exit 1; }
done
S=sbatch; [ "${DRY_RUN:-0}" = 1 ] && S="echo sbatch"
sub() { local id; id=$($S --parsable "$@") || { echo "❌ 제출 실패: $*" >&2; exit 1; }; echo "${id%%;*}"; }

JM=$(OPD_CKPT=$PWD/$OPD_CKPT OPD_OUT=$PWD/$OUT sub scripts/37_merge_opd.slurm)
JH=$(sub --dependency=afterok:$JM --time=03:00:00 \
  --export=ALL,EVAL_STAGES=init,INIT_MODEL=$PWD/$OUT,EVAL_N=all,EVAL_MAXTOK=8192,EVAL_PORT=8171,EVAL_RESULT_FILE=logs/eval_final_${TAG}_hold.jsonl \
  scripts/eval_midtrain.slurm)
JG=$(sub --dependency=afterok:$JM --time=04:00:00 \
  --export=ALL,EVAL_STAGES=sft,SFT_MODEL=$PWD/$OUT,EVAL_N=all,EVAL_MAXTOK=4096,EVAL_PORT=8172,EVAL_RESULT_FILE=logs/eval_final_${TAG}_gb.jsonl \
  scripts/eval_global_bench.slurm)
echo "merge=$JM holdout=$JH gbench=$JG"
cat <<EOF
끝난 뒤 짝지은 검정:
  python3 scripts/eval_paired.py logs/eval_items_init_83037.jsonl logs/eval_items_init_$JH.jsonl --by source
  python3 scripts/eval_paired.py logs/eval_items_gbench_sft_83038.jsonl logs/eval_items_gbench_sft_$JG.jsonl
EOF
