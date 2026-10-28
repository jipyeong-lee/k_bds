#!/bin/bash
# =============================================================================
# 36_opd_domain.slurm — OPD(온폴리시 증류) 도메인 라우팅 교사, KISTI 8gpu 노드의 GPU 1장.
#   학생 = merged_ta_w07(통합 모델), 교사 = sft_mixed_merged(SFT) + 전문가 어댑터 3개(deepvision 2192 · mmk12 350 · pmcvqa 4896).
#   쿼리의 이미지 경로(/images/<도메인>/)로 교사 어댑터를 고른다 → kgpu/gkd_domain_teacher.patch (ms-swift 4.1.3 GKD).
#   패치는 sandbox 를 건드리지 않고 swift 패키지 사본에 적용해 PYTHONPATH_FIRST 로 앞에 둔다(runc.sh 가 지원).
# 상태: 2026-10-28 초안. KISTI 에서 한 번도 실행하지 않았다(로컬 문법·스텁 시험만). 근거 측정은 kgpu(RTX PRO 6000 96GB):
#   deepvision 20 step 79 s/step · 최대 88.63 GiB, 실모델 라우팅 시험(T5) 통과.
# ⚠ 메모리: A100 80GB 의 사용 가능량은 약 79 GiB 라 위 설정 그대로는 초과할 것으로 계산했다(미측정). 그래서 OFFLOAD_TEACHER 기본값이 true 다
#   (교사 약 17.5 GiB 를 GPU 밖에 둔다). 도메인 라우팅과 오프로드의 조합은 시험한 적이 없다. 안 들어가면 VLLM_UTIL·MAX_COMP 를 낮출 것.
# ⚠ 노드: 8gpu 는 노드 배타 점유(시간당 8 노드시간)인데 GPU 는 1장만 쓴다(GKD+vLLM colocate 는 NPROC_PER_NODE=1).
# 사용:  sbatch scripts/36_opd_domain.slurm                                   # 20 step 스모크, 도메인마다 200행
#        DRY_RUN=1 srun -p cpu64-only -t 10 bash scripts/36_opd_domain.slurm  # 대기열 전에 1~3단계만 확인(GPU 불필요, 입력·패치·우선순위·프롬프트)
#        MAX_STEPS=100 SMOKE_ROWS=400 sbatch --time=06:00:00 scripts/36_opd_domain.slurm
# env(선택): MAX_STEPS(20) MAX_COMP(3072) SMOKE_ROWS(200, 도메인마다) GAS(4) OFFLOAD_TEACHER(true) VLLM_UTIL(0.35) BETA(1.0, 1=reverse KL)
#            OPD_MAX_PIXELS(262144) SAVE_STEPS(비면 저장 안 함) STUDENT TEACHER_BASE AD_DV AD_MM AD_PMC GKD_PATCH DRY_RUN
# 전제: 프로젝트 폴더에 runc.sh(PYTHONPATH_FIRST 지원판)와 kgpu/gkd_domain_teacher.patch 가 있을 것. 입력은 전부 로컬 경로다(계산 노드는 외부망이 없다).
# =============================================================================
#SBATCH --job-name=opd-domain
#SBATCH --partition=8gpu
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home01/k266a01/kbds_project/logs/opd_domain_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/opd_domain_%j.log

set -uo pipefail
cd "${PROJ_DIR:-/home01/k266a01/kbds_project}"
source scripts/00_common.sh
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
T0=$(date +%s); stamp(){ echo "[opd +$(( $(date +%s)-T0 ))s] $*"; }

STUDENT=${STUDENT:-$CKPT_DIR/merged_ta_w07}
TEACHER_BASE=${TEACHER_BASE:-$CKPT_DIR/sft_mixed_merged}
AD_DV=${AD_DV:-$WORK_DIR/b200_migrate/runs/deepvision_ep1_gdpo_async_tis_entmask/v38-20260821-064935/checkpoint-2192}
AD_MM=${AD_MM:-$CKPT_DIR/expert_mmk12_0821-1336/v0-20260821-202750/checkpoint-350}
AD_PMC=${AD_PMC:-$CKPT_DIR/expert_pmcvqa_0821-2028/v2-20260901-191946/checkpoint-4896}
MAX_STEPS=${MAX_STEPS:-20}; MAX_COMP=${MAX_COMP:-3072}; SMOKE_ROWS=${SMOKE_ROWS:-200}; GAS=${GAS:-4}
OUT=$CKPT_DIR/opd_domain_${SLURM_JOB_ID:-manual}; MIX=$OUT/opd_mix.jsonl

stamp "1) 입력 점검 — 학생 $STUDENT · 교사 $TEACHER_BASE"
for m in "$STUDENT" "$TEACHER_BASE"; do
  [ -f "$m/config.json" ] && [ -f "$m/model.safetensors.index.json" ] || { echo "❌ 모델 폴더 불완전: $m"; exit 1; }
done
for a in "$AD_DV" "$AD_MM" "$AD_PMC"; do
  [ -f "$a/adapter_model.safetensors" ] && [ -f "$a/adapter_config.json" ] || { echo "❌ 어댑터 파일 없음: $a"; exit 1; }
done
PATCH=${GKD_PATCH:-$PROJ_DIR/kgpu/gkd_domain_teacher.patch}
[ -f "$PATCH" ] || { echo "❌ 패치 파일 없음: $PATCH"; exit 1; }
command -v patch >/dev/null || { echo "❌ 호스트에 patch 명령이 없다"; exit 1; }

stamp "2) swift 사본에 도메인 라우팅 패치 적용(패치나 샌드박스의 swift 가 바뀌면 새 사본)"
SWIFT_SRC=$CONTAINER_IMG/usr/local/lib/python3.12/site-packages/swift
[ -f "$SWIFT_SRC/rlhf_trainers/gkd_trainer.py" ] && [ -f "$SWIFT_SRC/version.py" ] || { echo "❌ sandbox 에 gkd_trainer.py 또는 version.py 없음: $SWIFT_SRC"; exit 1; }
OVERLAY=$WORK_DIR/overlay/swift413_gkd_domain_$(cat "$PATCH" "$SWIFT_SRC/version.py" "$SWIFT_SRC/rlhf_trainers/gkd_trainer.py" | sha256sum | cut -c1-8)
if [ ! -f "$OVERLAY/.ok" ]; then
  TMPO=$OVERLAY.tmp.${SLURM_JOB_ID:-$$}
  { mkdir -p "$TMPO" && cp -r "$SWIFT_SRC" "$TMPO/swift" && patch -F0 --no-backup-if-mismatch -p1 -d "$TMPO" < "$PATCH" \
    && [ "$(grep -c _set_teacher_domain "$TMPO/swift/rlhf_trainers/gkd_trainer.py")" -ge 3 ] && touch "$TMPO/.ok"; } \
    || { echo "❌ 패치 적용 실패 — 원본 sandbox 는 그대로다"; rm -rf "$TMPO"; exit 1; }
  mv -T "$TMPO" "$OVERLAY" 2>/dev/null || rm -rf "$TMPO"   # 같은 시각에 다른 job 이 먼저 만들었으면 그 사본을 쓴다
fi
[ -f "$OVERLAY/.ok" ] || { echo "❌ 사본 없음: $OVERLAY"; exit 1; }
export PYTHONPATH_FIRST=$OVERLAY
SW=$(run_py python -c "import swift; print(swift.__file__)" | tail -n 1)   # stderr 는 작업 로그로 간다(import 오류 원인)
[ "$SW" = "$OVERLAY/swift/__init__.py" ] || { echo "❌ 패치한 swift 가 우선하지 않는다(위 import 오류, 또는 runc.sh 의 PYTHONPATH_FIRST 지원을 확인): [$SW]"; exit 1; }
echo "swift = $SW"

stamp "3) 프롬프트 — 도메인마다 앞 ${SMOKE_ROWS}행을 번갈아 섞고, 패치의 도메인 판별과 이미지 존재를 GPU 전에 검사"
mkdir -p "$OUT"
run_py python - "$DATA_DIR/domains" "$SMOKE_ROWS" "$MIX" <<'PY' || { echo "❌ 프롬프트 준비 실패"; exit 1; }
import json, os, sys
src, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
D = ("deepvision", "mmk12", "pmcvqa"); rows = {}; bad = []
for d in D:
    rows[d] = []
    with open(f"{src}/stage2_{d}.jsonl") as f:
        for line in f:
            if len(rows[d]) >= n: break
            rows[d].append(line.rstrip("\n"))
    if len(rows[d]) < n: sys.exit(f"{d}: {len(rows[d])}행뿐(요청 {n})")
    for i, line in enumerate(rows[d]):
        imgs = [p["path"] if isinstance(p, dict) else p for p in json.loads(line).get("images") or []]
        hits = [x for x in D if any(f"/{x}/" in p for p in imgs)]   # 패치의 판별식과 같다
        if hits != [d] or not all(os.path.exists(p) for p in imgs): bad.append((d, i, hits))
if bad: sys.exit(f"라우팅 불가 또는 이미지 없음 {len(bad)}행, 예: {bad[:3]}")
with open(out, "w") as f:
    for i in range(n):
        for d in D: f.write(rows[d][i] + "\n")
print(f"프롬프트 {3 * n}행(도메인당 {n}) → {out}")
PY
[ "${DRY_RUN:-0}" = 1 ] && { stamp "DRY_RUN=1 — 1~3단계 통과, 학습은 하지 않는다"; exit 0; }

stamp "4) swift rlhf gkd — ${MAX_STEPS} step, 학생 LoRA r32 ← 교사 SFT + 어댑터 3개 라우팅, lmbda 1.0(순수 온폴리시), 교사 오프로드 ${OFFLOAD_TEACHER:-true}"
export CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1   # GKD+vLLM colocate 는 프로세스 1개
export GKD_TEACHER_DOMAIN_ADAPTERS="deepvision=$AD_DV,mmk12=$AD_MM,pmcvqa=$AD_PMC"
SAVE=(--save_strategy no); [ -n "${SAVE_STEPS:-}" ] && SAVE=(--save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit 2)
run_py swift rlhf --rlhf_type gkd \
  --model "$STUDENT" --model_type "$MODEL_TYPE" --system "$SYSTEM_PROMPT" --dataset "$MIX" \
  --teacher_model "$TEACHER_BASE" \
  --lmbda 1.0 --beta "${BETA:-1.0}" --seq_kd false --sft_alpha 0 --temperature 1.0 \
  --offload_teacher_model "${OFFLOAD_TEACHER:-true}" \
  --tuner_type lora --lora_rank 32 --lora_alpha 32 --use_rslora true --lora_dropout 0 --target_modules all-linear --torch_dtype bfloat16 \
  --enable_thinking true --max_completion_length "$MAX_COMP" \
  --per_device_train_batch_size 1 --gradient_accumulation_steps "$GAS" --learning_rate 1e-5 \
  --max_length $(( 2048 + MAX_COMP )) --max_pixels "${OPD_MAX_PIXELS:-262144}" --gradient_checkpointing true --attn_impl "$ATTN_IMPL" \
  --use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size 1 --vllm_mm_processor_cache_gb 0 \
  --vllm_gpu_memory_utilization "${VLLM_UTIL:-0.35}" --vllm_max_model_len $(( 3072 + MAX_COMP )) --sleep_level 1 \
  --log_completions true --logging_steps 1 "${SAVE[@]}" --report_to none \
  --output_dir "$OUT" --max_steps "$MAX_STEPS"
TR=$?; echo "SWIFT_EXIT=$TR"

stamp "5) 판정 — 교사 도메인 카운터(teacher_steps/*)와 최대 메모리"
run_py python - "$OUT" "$MAX_STEPS" "$GAS" <<'PY'
import glob, json, sys
out, steps, gas = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
fs = glob.glob(f"{out}/*/logging.jsonl")
if len(fs) != 1: sys.exit(f"logging.jsonl {len(fs)}개(1개여야 한다)")
R = [r for r in map(json.loads, open(fs[0])) if "loss" in r]
if not R: sys.exit("loss 기록 없음")
D = ("deepvision", "mmk12", "pmcvqa"); cnt = {d: R[-1].get(f"teacher_steps/{d}") for d in D}
print(f"step 기록 {len(R)}/{steps} · loss {R[0]['loss']} → {R[-1]['loss']} · 최대 memory(GiB) {max(r.get('memory(GiB)', 0) for r in R)}")
print(f"교사 도메인 카운터(마이크로배치) {cnt} · 합 {sum(v or 0 for v in cnt.values())} · 기대 {steps * gas}")
ok = len(R) == steps and all(cnt.values()) and sum(cnt.values()) == steps * gas
print("OPD_DOMAIN", "PASS" if ok else "CHECK"); sys.exit(0 if ok else 1)
PY
SUM=$?
stamp "done (swift exit=$TR, 판정 exit=$SUM)"
exit $(( TR != 0 ? TR : SUM ))
