# 인수인계 (HANDOFF) — K-BDS 의료 멀티모달 파이프라인

> **갱신 2026-07-23.** 다른 계정/사람이 **이어받아 실행**하기 위한 단일 문서.
> 큰 그림·수치는 [`README.md`](README.md), 확장 재현은 [`docs/stage2_expansion_runbook.md`](docs/stage2_expansion_runbook.md).

---

## 1. 지금 어디까지 왔나 (한눈에)

| 단계 | 상태 | 핵심 |
|---|---|---|
| ① 콜드스타트 SFT | ✅ **v3 완료·평가됨** | 형식천장 완파(`format_think` v2 0.185→**v3 0.909**), 홀드아웃 acc v2 0.295→**v3 0.348** |
| ② 범용 RLVR | ✅ 방법론 종결(dr_grpo) · 🔧 **풀확장 세팅완료·실행대기** | DeepVision+MMK12+ThinkLite=**128,349**, init=v3 |
| ③ 의료 RL (RaR) | 🟡 배선 e2e PASS · **본실행 미시작** | judge(Qwen3.6-27B-FP8)·루브릭 검증완료 |
| ④ 평가 | 🔄 기준선·v3 홀드아웃 완료 | HealthBench base 0.229/v2 0.224(n=1000), v3 미측정 |

**바로 다음 임계경로**: Stage-2 풀확장 **배선 스모크 → 본실행**(다른 계정, ~70h/8gpu). 그 다음 Stage-3.

---

## 2. 🚨 다른 계정으로 옮길 때 — 반드시 먼저 (경로/토큰/컨테이너)

이 레포는 `/home01/k252a01/kbds_project` 를 가정한다. 새 계정에선 아래 3가지를 **먼저** 처리:

```bash
# (A) 경로 일괄 치환 — PROJ_DIR 은 env 로 되지만, *.slurm 의 #SBATCH --output 은 지시어라 치환 필요
NEW=/home01/<새계정>/kbds_project
grep -rl "/home01/k252a01/kbds_project" scripts/ | \
  xargs sed -i "s#/home01/k252a01/kbds_project#$NEW#g"
#   → 38개 스크립트의 하드코딩 경로 + 26개 slurm 의 --output 을 한 번에 교체.
#   00_common.sh 의 PROJ_DIR 도 이 값으로 바뀜(파생 WORK_DIR/HF_HOME/CKPT_DIR 자동 추종).

# (B) HF 토큰 (게이트·rate-limit 회피, MMK12 큰 parquet 정체 방지)
export HF_TOKEN=hf_xxx           # 이 계정은 ~/model_download.py 에 보유했음

# (C) 컨테이너 이미지 재빌드 (git 에 없음, ~수GB)
bash env/build_image.sh          # → $WORK_DIR/images/ms-swift-413-sandbox

# (D) 변환용 python (pyarrow+PIL) — 위 sed 가 못 잡는 계정밖 경로. 13_build_stage2_expanded 에서 씀.
export BUILD_PY=/home01/<새계정>/.conda/envs/<env>/bin/python   # pyarrow+PIL 있는 아무 python
```

> **git 에 없는 것(전부 재생성/재빌드)**: `work/` 하위 전체 — 데이터 jsonl·이미지·체크포인트·컨테이너·HF캐시. 코드/설정/문서만 git 에 있다. 데이터는 §4 로 재현.

---

## 3. 환경 함정 (겪고 기록한 것 — 재발 방지)

- **NVLink 없음**(PCIe A100, NCCL→SHM) → full-FT 375~660s/step. **전 단계 LoRA 필수**.
- **계산노드 컨테이너 불가**: CPU 노드 `max_user_namespaces=0` → Singularity 즉사. GPU 노드만 컨테이너 O. **데이터 변환은 conda `swift` env(pyarrow+PIL)로 CPU 잡** 실행(컨테이너 우회).
- **로그인노드 워치독**: 무거운 작업(대용량 이미지 추출 등) ~15분 무통보 kill. → 빌드는 CPU 잡으로. 로그는 항상 `python -u`.
- **HF 오프라인 플래그**: `00_common.sh` 가 `HF_HUB_OFFLINE=1` 설정 → 다운로드 시 런칭셸에서 **`unset HF_HUB_OFFLINE` + `--env HF_HUB_OFFLINE=0`**.
- **글로벌 vLLM 로그인노드 불가**(드라이버) → judge·학습·추론 전부 컴퓨트노드.
- **swift export 병합/eval 은 GPU 노드에서만**(컨테이너 필요). merge_lora 는 CPU 연산이지만 컨테이너가 CPU 노드서 안 도는 게 제약.

---

## 4. Stage-2 풀확장 — 실행 절차 (이 계정서 세팅·검증 완료)

전 과정 상세 = [`docs/stage2_expansion_runbook.md`](docs/stage2_expansion_runbook.md). 요약:

```bash
# 1) 데이터 다운로드 (로그인노드, 컨테이너, HF_TOKEN)
#    DeepVision-103K · medix-rl-data · FanqingM/MMK12 · russwang/ThinkLite-VL-hard-11k
# 2) 신규 2종 변환 (CPU 잡)
sbatch scripts/13_build_stage2_expanded.slurm
# 3) 확장셋 조립 (bytehash dedup, seed=42)
python3 scripts/build_stage2_mix.py            # → stage2_expanded_{train 128,349 / holdout 1,673}
# 4) v3 init 준비
sbatch scripts/10_sft.slurm                    # v3 SFT → sft_mixed_lora
sbatch scripts/12_merge_mixed.slurm            #  → sft_mixed_merged (또는 50_eval_v3.slurm)
# 5) Stage-2 확장 GRPO
SMOKE=1 bash scripts/launch_stage2_expanded.sh # 배선 스모크(먼저!)
bash scripts/launch_stage2_expanded.sh         # 본실행 (dr_grpo, ~70h)
# 6) 평가: stage2_expanded_holdout.jsonl 소스별(_source)·층별(_stratum) 채점
```

**검증된 기준선(이 계정 실측)**: v3 콜드스타트 홀드아웃 **0.348**(math 0.324=약점) / v2+RL 0.380~0.390.
→ 확장 성패 = **신규 홀드아웃(MMK12/ThinkLite)에서 v3(0.348) 대비 상승 + DeepVision 유지**.

---

## 5. 남은 일 (우선순위)

1. **Stage-2 확장 스모크 → 본실행** (다른 계정, 예산 5,000h).
2. **확장 결과 평가** → v3 대비 STEM 이득 확인.
3. **Stage-3 본실행**: `bash scripts/launch_stage3.sh` (init=Stage-2 산출 병합본, GDPO 레시피 권고). 계획서 핵심 산출물·미시작.
4. (옵션) v3 HealthBench Hard(n=1000, ~28 노드시간, gpu:2) — base 0.229/v2 0.224 대비.
5. (기록만) 구 DeepVision 홀드아웃 22% 오염 재구성.

---

## 6. 파일 지도 (핵심만)

```
scripts/
  00_common.sh                     공통 경로/환경/run_py 래퍼  ← 이식 시 PROJ_DIR
  10_sft.slurm                     Stage-1 v3 SFT (기본값=sft_mixed)
  build_mixed_coldstart.py         v3 콜드스타트 데이터 빌드
  50_eval_v3.slurm / eval_v3_holdout.py   병합+홀드아웃 평가(strict format_think·층별)
  52_eval_v2_baseline.slurm        v2 동일조건 재측정
  13_build_stage2_expanded.slurm   MMK12·ThinkLite 변환(CPU)
  build_stage2_mix.py              확장셋 조립+홀드아웃(bytehash dedup)
  convert_to_swift.py              parquet→swift (DeepVision/medix/MMK12/ThinkLite 자동감지)
  20_rlvr_grpo.slurm               Stage-2 GRPO (기본값=v3 init+확장셋)
  launch_stage2_expanded.sh        Stage-2 확장 표준 진입점
  30_medical_rl.slurm / launch_stage3.sh / judge_server.sh   Stage-3
  make_holdout.py                  DeepVision 층화 홀드아웃
configs/
  accuracy.py                      accuracy_mix + format_think 보상
  medical_reward.py                Stage-3 RaR clinical_judge
docs/
  stage2_expansion_runbook.md      확장 재현 0~6단계
  medical_reward_spec.md · worklog_*.md · project_status_*.md
work/  (git 제외 — 재생성)          data·images·checkpoints·hf_cache·images/컨테이너
```

**막히면**: README 현황·진행이력, 이 문서 §2(경로)·§3(함정) 순으로 확인.
