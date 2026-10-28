#!/usr/bin/env bash
# opd_smoke.sh — kgpu pod 에서 OPD(온폴리시 증류, ms-swift GKD) 배선 스모크. 학생=통합 모델 ta_w07, 교사=SFT+전문가 어댑터.
# 확인할 것: ① --reward_adapters 로 교사에 전문가 어댑터가 붙는가(--teacher_adapters 는 소스상 안 읽힌다, docs/21 4부 §3)
#           ② 96GB 에 학생+교사+vLLM colocate 가 들어가는가 ③ step 시간·메모리.
# 결과(2026-10-28): ① 확정 — 고정 응답 step 1 loss adapter 0.3856 대 control 0.7088, grad_norm 41.2 대 64.6, 메모리 차이 0.16 GiB(=어댑터 173MB).
#           ② 10 step 34 s/it · 최대 73.4 GiB(vLLM 포함). 판정은 반드시 FIXED_JSON 모드로 한다(아래 TOP_K 주석 참고).
#           ③ deepvision 20 step(PROMPTS=dv, MAX_COMP 4096): step 2~20 평균 79 s(step 1 은 110 s) · 최대 88.63 GiB(step 18, 96GB 의 93%).
#              ta_w07 응답 80개: 평균 1,548 · 최대 3,068 토큰, 상한(4096) 도달 0개, </answer> 없음 1개(1,313 토큰, 형식 실패).
#              상한 길이의 응답은 나오지 않아 그 길이의 메모리는 모른다. mmk12·pmcvqa 응답 길이도 아직 재지 않았다.
# 전제(~/files = /root/files, 영속): kbds/{code.tgz, setup_venv.sh, sft_mixed_merged/, ckpt_dv.tar(=checkpoint-2192 어댑터)}
# env(필수): HF_TOKEN(pod env, 학생 모델이 비공개 저장소). run_pod.sh 로: bash kgpu/run_pod.sh opd_smoke.sh HF_TOKEN
# env(선택): MAX_STEPS(10) · MAX_COMP(2048, PROMPTS=dv 면 3072) · TEACHER_TAG(dv) · VLLM_UTIL(0.35) · OFFLOAD_TEACHER(false) · BETA(1.0, 0=forward KL 1=reverse KL)
#           PROMPTS=dv → deepvision 프롬프트. ~/files/kbds/opd_dv/{stage2_deepvision.jsonl, dv_images.tar} 의 앞 SMOKE_ROWS(200)행을 쓰고,
#             jsonl 의 KISTI 절대경로를 pod 로컬 경로로 바꾸고 그 행이 쓰는 이미지만 tar 에서 푼다. 긴 응답에서의 메모리 확인용이다.
#             dv 의 MAX_COMP 3072 는 위 결과 ③ 의 최대(3,068)에 맞춘 값이다. 이 값으로는 아직 실행하지 않았다(③ 은 4096 으로 돌렸다).
#             (docs/00_overview 의 평균 토큰 4,184 / 702 / 872 는 도메인이 아니라 기본 모델 / SFT 모델 / ta_w07 의 평균이다.)
#             결과 ③ 재현: bash kgpu/run_pod.sh opd_smoke.sh HF_TOKEN PROMPTS=dv MAX_STEPS=20 MAX_COMP=4096 RUN_TAG=dv20_repro
#               ③ 의 원자료(응답 80개·step 지표)는 kbds/logs/opd_runs_adapter_dv/{completions,logging}.jsonl 이고 로그는 kbds/logs/opd_smoke_adapter_dv_1028-0608.log 다.
#               opd_runs_<RUN_TAG>/ 에 파일이 이미 있으면 시작 직후 ❌ 로 종료한다(원자료 덮어쓰기 방지, 빈 폴더는 통과). RUN_TAG 를 빼면 기본 태그 adapter_dv 가 ③ 의 폴더와 같아 막히므로 재현은 다른 태그로 한다.
#               기본 MAX_COMP 는 3072 라서 ③ 과 같은 4096 은 명시해야 한다.
#           TOP_K=1 → 학생 롤아웃을 그리디로 만든다. 그래도 두 실행의 응답이 같아지지는 않았다(근접 동률 토큰에서 갈리고 뒤로 갈수록 길이가 달라진다).
#             그래서 loss 차이를 교사 탓으로 돌리는 판정에는 쓸 수 없다. 판정은 FIXED_JSON 으로.
#           FIXED_JSON=<~/files/kbds 아래 파일> → 고정 응답 모드. [{"user","completion"}] 로 만든 데이터셋을 --lmbda 0(샘플링 없음)·vLLM 끄고 돌린다.
#             학생·토큰이 두 실행에서 완전히 같아서 loss 차이는 교사(어댑터 유무)만 반영한다. MAX_STEPS=1 · 응답 4개 = 1 step.
#           NO_ADAPTER=1 → --reward_adapters 를 뺀 대조 실행(RUN_TAG=control). FIXED_JSON 과 같은 설정으로 두 번 돌려 step 1 loss 를 비교한다.
#           DOMAIN_ROUTE=1 → (FIXED_JSON 필요, T5) kgpu/gkd_domain_teacher.patch 를 실모델로 시험한다. 고정 응답 4개를 /deepvision/·/mmk12/·/pmcvqa/ 폴더로 복사해
#             12 step 을 patched 교사(어댑터 3개 dv 2192·mm 350·pmc 4896)로 한 번, 같은 4개를 --reward_adapters 한 개로만 돌린 기준 실행을 도메인마다 한 번(4 step) 돌린다.
#             학습률 0·GAS 1·셔플 끔이라 step 별 loss 가 (행, 교사) 만의 함수다. 끝의 판정 블록이 C0 step 수 · C1 도메인 카운터 · C2 routed = 단일 교사(오차 1%) · C3 기준 실행끼리 3% 이상 다름을 낸다.
#             전제: ~/files/kbds/gkd_domain_teacher.patch · ckpt_mm350.tar · ckpt_pmc.tar. bash kgpu/run_pod.sh opd_smoke.sh HF_TOKEN FIXED_JSON=fixed_step1.json DOMAIN_ROUTE=1 RUN_TAG=t5 MAX_SEC=3600
# PROMPTS 기본값은 배선 확인용 medix 200행이다. mmk12/pmcvqa 모드는 아직 없다(mmk12 데이터는 kbds/mmk12_data.tar 에 있고, pmcvqa 프롬프트·이미지는 kgpu 에 없다).
set -uo pipefail
F=/root/files/kbds; W=/root/w; VENV=/root/venv; RUN_TAG=${RUN_TAG:-adapter${PROMPTS:+_$PROMPTS}}; LOG=$F/logs/opd_smoke_${RUN_TAG}_$(date +%m%d-%H%M).log
mkdir -p "$W" "$F/logs"; exec > >(tee "$W/smoke.log") 2>&1
trap 'cp "$W/smoke.log" "$LOG"' EXIT   # 실패해도 로그를 ~/files 로 남긴다
[ -n "$(ls -A "$F/logs/opd_runs_${RUN_TAG}" 2>/dev/null)" ] && { echo "❌ opd_runs_${RUN_TAG} 에 이미 파일이 있다 — RUN_TAG 를 바꿀 것(원자료 덮어쓰기 방지)"; exit 1; }   # GPU 작업 전에 막는다. 빈 폴더(실패한 실행의 흔적)는 통과
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTORCH_ALLOC_CONF=expandable_segments:True
TAG=${TEACHER_TAG:-dv}
PROMPTS=${PROMPTS:-medix}; [ "$PROMPTS" = dv ] && : "${MAX_COMP:=3072}"; : "${MAX_COMP:=2048}"
case $PROMPTS in medix|dv) ;; *) echo "❌ PROMPTS=$PROMPTS (medix|dv)"; exit 1;; esac
[ "$PROMPTS" = dv ] && [ -n "${FIXED_JSON:-}" ] && { echo "❌ FIXED_JSON 은 medix 전용"; exit 1; }
nvidia-smi -L

stamp "1) code + 교사 베이스(네트워크 공유라 느림 → venv 설치와 병행) + 전문가 어댑터"
tar xzf "$F/code.tgz" -C "$W" --no-same-owner || { echo "❌ code.tgz"; exit 1; }
[ -d "$W/teacher_base" ] || { cp -r "$F/sft_mixed_merged" "$W/teacher_base" & CP=$!; }
mkdir -p "$W/ck_$TAG" && tar xf "$F/ckpt_$TAG.tar" -C "$W/ck_$TAG" --no-same-owner || { echo "❌ ckpt_$TAG"; exit 1; }
ADAPTER=$(ls -d "$W"/ck_$TAG/checkpoint-* | head -1); ls "$ADAPTER"
[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "❌ adapter_model.safetensors 없음: '$ADAPTER'"; exit 1; }
ADAPTER_ARG=(--reward_adapters "$ADAPTER"); [ "${NO_ADAPTER:-0}" = 1 ] && ADAPTER_ARG=()   # swift 는 교사 어댑터를 reward_adapters 에서 읽는다
if [ "${DOMAIN_ROUTE:-0}" = 1 ]; then   # T5: 어댑터 3개를 풀고 GKD_TEACHER_DOMAIN_ADAPTERS 값을 만든다(머리말)
  { [ -n "${FIXED_JSON:-}" ] && [ "${NO_ADAPTER:-0}" != 1 ] && [ "$TAG" = dv ]; } || { echo "❌ DOMAIN_ROUTE 는 FIXED_JSON 필요 · NO_ADAPTER 불가 · TEACHER_TAG=dv"; exit 1; }
  [ -f "$F/gkd_domain_teacher.patch" ] || { echo "❌ ~/files/kbds/gkd_domain_teacher.patch 없음"; exit 1; }
  declare -A AD=([deepvision]=$ADAPTER)
  for p in mmk12:mm350 pmcvqa:pmc; do
    t=${p#*:}; mkdir -p "$W/ck_$t" && tar xf "$F/ckpt_$t.tar" -C "$W/ck_$t" --no-same-owner || { echo "❌ ckpt_$t"; exit 1; }
    AD[${p%%:*}]=$(ls -d "$W"/ck_$t/checkpoint-* | head -1)
  done
  for d in "${!AD[@]}"; do [ -f "${AD[$d]}/adapter_model.safetensors" ] && [ -f "${AD[$d]}/adapter_config.json" ] || { echo "❌ $d 어댑터 파일 없음: '${AD[$d]}'"; exit 1; }; done
  SPEC="deepvision=${AD[deepvision]},mmk12=${AD[mmk12]},pmcvqa=${AD[pmcvqa]}"; echo "teacher adapters: $SPEC"
fi

stamp "2) venv"
. "$F/setup_venv.sh"
if [ "${DOMAIN_ROUTE:-0}" = 1 ]; then   # 설치된 ms-swift(4.1.3 핀)에 패치를 적용한다. 컨텍스트가 어긋나면 patch 가 실패하고 여기서 멈춘다
  SP=$($PY -c "import importlib.util as u,os;print(os.path.dirname(u.find_spec('swift').submodule_search_locations[0]))")
  command -v patch >/dev/null || $APT install -y -qq patch >/dev/null
  patch -p1 -d "$SP" < "$F/gkd_domain_teacher.patch" || { echo "❌ gkd_domain_teacher.patch 적용 실패"; exit 1; }
fi

stamp "2b) 학생 ← HF easypyeong/Qwen3.5-9B-MedCrossReason (ta_w07)"
$PY -c "from huggingface_hub import snapshot_download as s;s('easypyeong/Qwen3.5-9B-MedCrossReason',local_dir='$W/student')" || { echo "❌ student download"; exit 1; }
[ -n "${CP:-}" ] && { wait $CP || { echo "❌ teacher_base copy"; exit 1; }; }
[ -f "$W/teacher_base/config.json" ] || { echo "❌ teacher_base"; exit 1; }

stamp "3) $PROMPTS 프롬프트 ${SMOKE_ROWS:=200}행"
DATASET=$W/data/medix.jsonl
if [ "$PROMPTS" = dv ]; then
  DATASET=$W/data/dv.jsonl; IMG=$W/data/images/deepvision; mkdir -p "$IMG"
  head -n "$SMOKE_ROWS" "$F/opd_dv/stage2_deepvision.jsonl" | sed "s#/home01/k266a01/kbds_project/work/data/images/deepvision/#$IMG/#g" > "$DATASET"
  grep -oE "$IMG/[^\"]+" "$DATASET" | sed "s#^$IMG/##" > "$W/dv_need.txt"
  [ -s "$W/dv_need.txt" ] || { echo "❌ dv 프롬프트가 참조하는 이미지가 없다(jsonl 이 비었거나 경로 접두사가 다르다). 빈 목록이면 tar 가 전부 푼다"; exit 1; }
  tar xf "$F/opd_dv/dv_images.tar" -C "$IMG" -T "$W/dv_need.txt" --no-same-owner || { echo "❌ dv_images.tar"; exit 1; }
  miss=0; while IFS= read -r f; do [ -f "$IMG/$f" ] || miss=$((miss+1)); done < "$W/dv_need.txt"
  left=$(grep -c /home01/ "$DATASET")
  echo "dv 이미지 $(wc -l < "$W/dv_need.txt")개 · 없는 파일 $miss · KISTI 경로가 남은 행 $left"
  { [ "$miss" = 0 ] && [ "$left" = 0 ]; } || { echo "❌ dv 프롬프트 준비 실패"; exit 1; }
elif [ ! -f "$W/data/medix.jsonl" ]; then
  $PY -c "from huggingface_hub import snapshot_download as s;print(s('MBZUAI/medix-rl-data',repo_type='dataset',allow_patterns=['data/train-*']))" > "$W/mx_path"
  $PY "$W/scripts/convert_to_swift.py" MBZUAI/medix-rl-data --parquet "$(tail -1 "$W/mx_path")"/data/train-*.parquet \
    --out "$W/data/medix.jsonl" --images-dir "$W/data/images" --limit "$SMOKE_ROWS" || { echo "❌ convert"; exit 1; }
fi
wc -l "$DATASET"

LMBDA=1.0; USE_VLLM=true; VLLM_MODE="--vllm_mode colocate"
if [ -n "${FIXED_JSON:-}" ]; then
  $PY - "$F/$FIXED_JSON" "$W/data/medix.jsonl" "$W/data/fixed.jsonl" <<'PY' || { echo "❌ fixed dataset"; exit 1; }
import json, sys
src, med, out = sys.argv[1:4]
rows = [json.loads(l) for l in open(med)]
items = json.load(open(src)); n = 0
with open(out, "w") as f:
    for it in items:
        u = it["user"].strip()
        r = next((r for r in rows if r["messages"][0]["content"].strip() == u), None) or \
            next((r for r in rows if u in r["messages"][0]["content"] or r["messages"][0]["content"].strip() in u), None)
        if r is None: sys.exit("medix 에서 못 찾음: " + u[:80])
        f.write(json.dumps({"messages": [{"role": "user", "content": r["messages"][0]["content"]},
                                         {"role": "assistant", "content": it["completion"]}],
                            "images": r["images"], "solution": r.get("solution", "")}, ensure_ascii=False) + "\n"); n += 1
print("fixed rows:", n)
PY
  if [ "${DOMAIN_ROUTE:-0}" = 1 ]; then   # 같은 행을 도메인 폴더로 복사 → routed.jsonl(12행, 도메인 순서) + ref_<도메인>.jsonl(4행)
    $PY - "$W/data/fixed.jsonl" "$W/data" <<'PY' || { echo "❌ routed dataset"; exit 1; }
import json, os, shutil, sys
src, root = sys.argv[1:3]
rows = [json.loads(l) for l in open(src)]
allf = open(f"{root}/routed.jsonl", "w")
for d in ("deepvision", "mmk12", "pmcvqa"):
    with open(f"{root}/ref_{d}.jsonl", "w") as f:
        for i, r in enumerate(rows):
            imgs = []
            for j, p in enumerate(r["images"]):
                p = p["path"] if isinstance(p, dict) else p
                dst = f"{root}/domain_imgs/{d}/{i}_{j}_{os.path.basename(p)}"
                os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(p, dst); imgs.append(dst)
            line = json.dumps({**r, "images": imgs}, ensure_ascii=False) + "\n"
            f.write(line); allf.write(line)
print("routed rows:", 3 * len(rows))
PY
  fi
  DATASET=$W/data/fixed.jsonl; LMBDA=0; USE_VLLM=false; VLLM_MODE=""   # ms-swift 는 use_vllm=false 일 때 --vllm_mode 를 거부한다(rlhf_args.py:393)
fi

stamp "4) swift rlhf gkd — ${MAX_STEPS:=10} step, 학생 ta_w07(LoRA r32) ← 교사 SFT+$TAG 어댑터, lmbda 1.0(순수 온폴리시)"
SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."
gkd(){   # $1=데이터셋 $2=step 수 $3=출력 하위폴더, 나머지는 swift 에 그대로 넘긴다. GAS·LR env 로 학습 설정을 바꾼다
  local ds=$1 ms=$2 out=$3; shift 3
  NPROC_PER_NODE=1 "$VENV/bin/swift" rlhf --rlhf_type gkd \
    --model "$W/student" --model_type qwen3_5 --system "$SYS" --dataset "$ds" ${FIXED_JSON:+--split_dataset_ratio 0} \
    --teacher_model "$W/teacher_base" "$@" \
    --lmbda $LMBDA --beta ${BETA:-1.0} --seq_kd false --sft_alpha 0 --temperature 1.0 ${TOP_K:+--top_k $TOP_K} \
    --offload_teacher_model ${OFFLOAD_TEACHER:-false} \
    --tuner_type lora --lora_rank 32 --lora_alpha 32 --use_rslora true --lora_dropout 0 --target_modules all-linear --torch_dtype bfloat16 \
    --enable_thinking true --max_completion_length $MAX_COMP \
    --per_device_train_batch_size 1 --gradient_accumulation_steps ${GAS:-4} --learning_rate ${LR:-1e-5} \
    --max_length $(( 2048 + MAX_COMP )) --max_pixels 262144 --gradient_checkpointing true --attn_impl sdpa \
    --use_vllm $USE_VLLM $VLLM_MODE --vllm_tensor_parallel_size 1 --vllm_mm_processor_cache_gb 0 \
    --vllm_gpu_memory_utilization ${VLLM_UTIL:-0.35} --vllm_max_model_len $(( 3072 + MAX_COMP )) --sleep_level 1 \
    --log_completions true --logging_steps 1 --save_strategy no --report_to none \
    --output_dir "$W/runs/$out" --max_steps $ms
}
if [ "${DOMAIN_ROUTE:-0}" = 1 ]; then
  RX=(--train_dataloader_shuffle false)
  GAS=1 LR=0 GKD_TEACHER_DOMAIN_ADAPTERS="$SPEC" gkd "$W/data/routed.jsonl" 12 route_routed "${RX[@]}"; RC=$?
  for d in deepvision mmk12 pmcvqa; do
    GAS=1 LR=0 gkd "$W/data/ref_$d.jsonl" 4 "route_ref_$d" "${RX[@]}" --reward_adapters "${AD[$d]}" || RC=$?
  done
  echo "SWIFT_EXIT=$RC"
  grep -iE "missing (adapter )?keys|unexpected keys" "$W/smoke.log" | cut -c1-200 | head -n 5
  mkdir -p "$F/logs/opd_runs_${RUN_TAG}"; for f in "$W"/runs/route_*/*/logging.jsonl; do cp "$f" "$F/logs/opd_runs_${RUN_TAG}/$(basename "$(dirname "$(dirname "$f")")")_logging.jsonl"; done
  stamp "[$RUN_TAG] T5 판정"
  $PY - "$W/runs" <<'PY'
import glob, json, sys
doms = ("deepvision", "mmk12", "pmcvqa")
def steps(tag):
    fs = glob.glob(f"{sys.argv[1]}/{tag}/*/logging.jsonl")
    if len(fs) != 1: sys.exit(f"❌ {tag}: logging.jsonl {len(fs)}개(1개여야 한다)")
    return [r for r in map(json.loads, open(fs[0])) if "loss" in r]
R = steps("route_routed"); ref = {d: steps(f"route_ref_{d}") for d in doms}
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(("✅ " if cond else "❌ ") + name, info)
rel = lambda x, y: abs(x - y) / max(abs(y), 1e-9)
check("C0 step 수 12/4/4/4", len(R) == 12 and all(len(v) == 4 for v in ref.values()), f"{len(R)} {[len(v) for v in ref.values()]}")
prev = dict.fromkeys(doms, 0); seq = []
for r in R:
    cur = {d: r.get(f"teacher_steps/{d}", 0) for d in doms}
    seq.append([d for d in doms if cur[d] - prev[d] == 1]); prev = cur
check("C1 step 별 교사 도메인 = 데이터 순서(4/4/4)", seq == [[d] for d in doms for _ in range(4)], str(seq))
for i, d in enumerate(doms):
    a, b = [r["loss"] for r in R[4*i:4*i+4]], [r["loss"] for r in ref[d]]
    check(f"C2 {d}: routed = 단일 교사(오차 1%)", len(a) == len(b) == 4 and all(rel(x, y) < 0.01 for x, y in zip(a, b)), f"routed {a} ref {b}")
sens = {(d, e): max(rel(x["loss"], y["loss"]) for x, y in zip(ref[d], ref[e])) for d in doms for e in doms if d < e}
check("C3 기준 실행끼리 3% 이상 다름(아니면 C2 는 의미 없음)", all(v > 0.03 for v in sens.values()), str({f"{d}/{e}": round(v, 3) for (d, e), v in sens.items()}))
print("T5", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
  echo "T5_EXIT=$?"
else
  gkd "$DATASET" "$MAX_STEPS" opd_smoke "${ADAPTER_ARG[@]}"
  echo "SWIFT_EXIT=$?"
  stamp "[$RUN_TAG] step 별 지표 — adapter 실행과 control 실행의 step 1 loss 를 비교한다"
  tr '\r' '\n' < "$W/smoke.log" | grep -E "^\{'loss'" | cut -c1-300 | head -n "$MAX_STEPS"
fi
# 학생 응답이 두 실행에서 같은지 비교하려면 jsonl 이 필요하다(작은 파일만 복사)
mkdir -p "$F/logs/opd_runs_${RUN_TAG}"; find "$W/runs/opd_smoke" -name '*.jsonl' -size -50M -exec cp {} "$F/logs/opd_runs_${RUN_TAG}/" \; 2>/dev/null
[ -n "$(ls -A "$F/logs/opd_runs_${RUN_TAG}")" ] || echo "⚠️ jsonl 없음 — 학생 응답 비교 불가(loss 만 비교할 수 있다)"
ls -l "$F/logs/opd_runs_${RUN_TAG}" | tail -n +2
stamp "done"
