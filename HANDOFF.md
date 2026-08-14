# 인수인계 (HANDOFF) — K-BDS 의료 멀티모달 파이프라인

> **갱신 2026-08-14.** 다른 계정/사람이 **이어받아 실행**하기 위한 단일 문서.
> 큰 그림·수치는 [`README.md`](README.md), 확장 재현은 [`docs/stage2_expansion_runbook.md`](docs/stage2_expansion_runbook.md).

---

## 0. 🆕 B200 플랫폼으로 이관 (2026-08-14) — 파이프라인 실행 검증됨

KISTI 노드 시간이 종료되어 **Jukyung-Yadok(NHN) 8× B200** 플랫폼으로 옮겼다. 실행 방법·함정은
[`CLAUDE.md`](CLAUDE.md) §1, 스크립트는 [`b200/`](b200/). 요지:

- **하드웨어**: 8× B200 180GB NVLink (KISTI A100-40GB·NVLink無 대비 대폭 상향). GRPO **~40 s/it**(스모크) vs KISTI 213.
- **접속**: HTTP API + PAT(`$ORCH_PAT`). 세션 생성→`exec`→job 폴링(대화형 셸 없음). `b200/drive_node.sh`로 감쌈.
- **저장소**: `$ORCH_HOME`(xfs 3.2T, job R/W) — venv·데이터·체크포인트 전부 여기. 세션 워크스페이스·`/tmp`는 휘발성.
- **반입**: KISTI→플랫폼 `/me/data/upload` ~138 MB/s, **단일 요청 60초 제한 → 3.5G 청크**. HF 대용량 CDN은 차단(직접 다운로드 불가).
- **스택**: KISTI 검증본 핀(ms-swift 4.1.3 / torch 2.10.0+cu129 / vllm 0.19.1 …). B200 5대 함정(uv 인덱스·cutlass-dsl 4.5.0.dev0·Triton 캐시 exec·flashinfer nvrtc.h·attn sdpa) 전부 `b200/node_setup_and_smoke.sh`에 반영.

**✅ 검증(2026-08-14)**: pmcvqa 3-step 스모크 완주 — `SWIFT_EXIT=0`, 보상 3종 정상,
mem 172/180 GiB, KL·grad 정상. 데이터 로드→이미지→vLLM 롤아웃(flashinfer)→보상(math_verify)→GDPO 손실→저장 전 구간 동작.

**다음 임계경로 (B200)**:
1. **deepvision·mmk12 데이터 업로드** — 현재 pmcvqa만 올라감. `data.tar`(46G, KISTI `/scratch/migrate_k266_to_gpu/`)를 `b200/upload_chunked.sh`로 청크 업로드.
2. **3-arm 배선** — `node_setup_and_smoke.sh`의 `swift rlhf`를 arm별로(`--dataset stage2_<arm>.jsonl`, `--max_steps 1500`, `--output_dir runs/expert_<arm>`) 복제. 8 GPU 배치(tensor-parallel vs arm 순차) 결정. 인자 근거 = `scripts/launch_domain_experts.sh`.
3. **어댑터 통합 → Stage-3** (아래 §5와 동일, 계획서 핵심 산출물·미시작).

> ⚠️ KISTI 원본(`k266a01:~/kbds_project/work`, ~305G)과 검증본은 그대로 보존. B200은 사본이므로
> KISTI 결과 재현이 목표 — 그래서 스택을 KISTI 버전으로 핀했다.

---

## 1. 지금 어디까지 왔나 (한눈에)

| 단계 | 상태 | 핵심 |
|---|---|---|
| ① 콜드스타트 SFT | ✅ **v3 완료·평가됨** | 형식천장 완파(`format_think` v2 0.185→**v3 0.909**), 홀드아웃 acc v2 0.295→**v3 0.348**. `sft_mixed_merged` = 이후 전 단계의 init |
| ② 범용 RLVR | 🔄 **도메인 전문가 3종 제출·큐 대기** | 구 혼합 실행은 step ~900 **형식 붕괴**로 취소. 최고점 **ck-850(+8.18pp, p<0.0001)** 스냅샷 확보 → 3분할 재설계 후 재제출(75394/75395/75396) |
| ③ 의료 RL (RaR) | 🟡 배선 e2e PASS · **본실행 미시작** | judge(Qwen3.6-27B-FP8)·루브릭 검증완료(유닛 29/29). **계획서 핵심 산출물인데 아직 안 돌았다** |
| ④ 평가 | 🔄 기준선 확보 | HealthBench Hard base 0.229 / v2 0.224(n=1000), v3 미측정 |

**바로 다음 임계경로**: 전문가 3종 완주(예상 시작 08-17) → 어댑터 통합 → **Stage-3 본실행**.

⚠️ **예산**: 5,000 노드시간 중 **874 집행 · 잔여 4,126**.
전문가 3종(1,500 step × 3)이 **2,130 = 잔여의 52%**, **Stage-3·평가에 남는 건 1,996**.

### 실행 중인 잡 (2026-08-14 제출)

| job | arm | 데이터 | 출력 |
|---|---|---|---|
| **75394** | deepvision | 40,000 | `expert_deepvision_0814-0812` |
| **75395** | mmk12 | 15,204 | `expert_mmk12_0814-0812` |
| **75396** | pmcvqa | 19,583 | `expert_pmcvqa_0814-0812` |

공통: `RECIPE=stable` · `NUM_GEN=8` · `MAX_STEPS=1500` · `LORA_DROPOUT=0` · `WATCHDOG=1` · init `sft_mixed_merged` · walltime 118h · 89 벽시계h/arm.
노드가 정확히 3개다 — **3 arm 동시 제출 = 파티션 전체 점유**.

### 🚨 인수인계 시 반드시 알아야 할 사건 — Stage-2 형식 붕괴

구 혼합 실행(73924/73925)이 **step ~900 에서 붕괴**했다. 이 프로젝트의 모든 현재 설계는 여기서 나온다.

- **step 850 이 최고점**이었다 — 홀드아웃 51.52%, init 대비 **+8.18pp(p<0.0001)**. 붕괴 직전까지 정상 개선 중이었다.
- 붕괴는 3단계 — **① 형식 무너짐(899~904) → ② 10 step 뒤 길이 폭주 → ③ `overlong_filter` 가 회복 차단.**
- **개시 원인은 미규명**이고 **28/28 동일 조건 재현 실패** → 결정론적 원인이 아니다. 원인 규명은 **포기**했고 증폭 경로 차단으로 방향을 틀었다.
- 붕괴 후 **13h35m(109 GPU-h)을 더 돌았다** — 볼 장치가 없었기 때문. 그래서 지금은 `WATCHDOG=1` 이 기본이다.
- ⚠️ **재개하지 말 것.** 재시작점 step 700 권고는 폐기됐다. `sft_mixed_merged` 에서 새로 시작한다.
- 전량 근거 → [`docs/stage2_run73924_postmortem.md`](docs/stage2_run73924_postmortem.md) (rev.2 에서 초판 3건을 정정했다)

**같이 확인된 것**: 의료(pmcvqa)는 RL 여섯 지점 전부 무변화(p>0.5). 붕괴와 무관하게 성립하고, **Stage-3 의 존재 이유를 굳힌다.**

---

## 2. 🚨 다른 계정으로 옮길 때 — 반드시 먼저 (경로/토큰/컨테이너)

이 레포는 현재 `/home01/k266a01/kbds_project` 를 가정한다. **이관은 이미 두 번 있었다**(k252a01 → k252a02 → k266a01).
새 계정에선 아래를 **먼저** 처리:

```bash
# (A) 경로 일괄 치환 — PROJ_DIR 은 env 로 되지만, *.slurm 의 #SBATCH --output 은 지시어라 치환 필요
NEW=/home01/<새계정>/kbds_project
grep -rl "/home01/k266a01/kbds_project" scripts/ | \
  xargs sed -i "s#/home01/k266a01/kbds_project#$NEW#g"
#   → 스크립트의 하드코딩 경로 + slurm 의 --output 을 한 번에 교체.
#   00_common.sh 의 PROJ_DIR 도 이 값으로 바뀜(파생 WORK_DIR/HF_HOME/CKPT_DIR 자동 추종).
#   ⚠️ 이 sed 는 무차별이다. 아래 두 곳은 치환하면 안 되니 끝나고 반드시 확인:
#      - transfer_pull.sh 의 SRC (소스 계정이어야 한다. SRC==DST 면 스크립트가 막는다)
#      - 32_net_test.slurm / 13_build_stage2_expanded.slurm 의 PY (계정밖 conda env → (D) 참조)

# (B) HF 토큰 (게이트·rate-limit 회피, MMK12 큰 parquet 정체 방지)
export HF_TOKEN=hf_xxx

# (C) 컨테이너 이미지 재빌드 (git 에 없음, ~수GB)
bash env/build_image.sh          # → $WORK_DIR/images/ms-swift-413-sandbox
#  ⚠️ 2026-07-27 이후 클러스터 apptainer 가 파손돼 이 빌드도 불가(singularity pull/build 필요).
#     이미지를 §2-A 직접전송으로 받아오고, 실행은 ENV_MODE=loader 우회 사용. → §3 맨 아래

# (D) 변환용 python (pyarrow+PIL) — 위 sed 가 못 잡는 계정밖 경로.
#     13_build_stage2_expanded.slurm · 32_net_test.slurm 이 쓴다. 둘 다 BUILD_PY 로 덮인다.
export BUILD_PY=/home01/<새계정>/.conda/envs/<env>/bin/python   # pyarrow+PIL 있는 아무 python
```

> **git 에 없는 것**: `work/` 하위 전체(데이터·이미지·체크포인트·컨테이너·HF캐시). 코드/설정/문서만 git.
> **두 경로**: ⓐ **직접 전송**(같은 클러스터 — 아래) 이 재다운로드/재빌드보다 훨씬 빠름. ⓑ 다른 클러스터면 §4 재현.

### 2-A. 직접 전송 (같은 클러스터·같은 그룹 — 권장)

`work/` 데이터를 재생성하지 말고 소스 계정에서 **당겨온다**. 확인된 사실(2026-07-24): 같은 그룹(`kbds0754`) 계정끼리
소스 홈이 group-readable(`drwxr-x---`)·파일 644 → **받는 계정이 pull 가능**(소스는 남의 홈에 push 불가).

**⚠️ 순서가 있다. 코드·문서는 이 스크립트로 옮기지 말 것 — git 이 정본이다.**

```bash
# ① 먼저 clone (코드·문서·설정). transfer_pull 이 scripts/ 경로 치환을 하므로 이게 먼저다.
git clone https://github.com/jipyeong-lee/k_bds.git kbds_project && cd kbds_project

# ② 그 다음 work/ 만 채운다 — SRC 를 반드시 지정한다:
SRC=/home01/<소스계정>/kbds_project bash scripts/transfer_pull.sh                # Stage-2 필수 subset
SRC=... WITH_STAGE3=1 bash scripts/transfer_pull.sh    # + judge(27B) + medix
SRC=... WITH_CKPT=1   bash scripts/transfer_pull.sh    # + ck-850 (구 실행 최고점 — 재생성 불가)
SRC=... WITH_LOGS=0   bash scripts/transfer_pull.sh    # 학습 로그 제외(기본은 포함, ~142M)
```
- 전송량 ≈ **60~80GB**(데이터+이미지 + `sft_mixed_merged` 18G + base 9B + 컨테이너). 전체 work/(~210G)는 불필요.
- **⚠️ 핵심**: jsonl 의 이미지 경로가 **절대경로** → 전송 후 반드시 치환. 스크립트가 `scripts/` + `work/data/` **전체**를 sed 처리한다(예전엔 `work/data/*.jsonl` 만 훑어 `domains/` 하위가 통째로 빠졌다 — 학습이 이미지를 못 찾고 죽는다).
- **`WITH_CKPT=1` 을 빠뜨리지 말 것**: ck-850 은 구 실행 최고점이고 **재생성 불가**다. 전문가 3분할에서 E1 을 빼고 이걸 일반 교사로 쓰는 선택지가 여기 달려 있다.
- **`logs/` 는 기본 포함**(142M). 사후분석·붕괴 진단·감시자 임계값이 전부 이 로그에서 나왔고 `watch_format_collapse.py --simulate` 가 이걸 먹는다.
- SRC 를 안 주면 직전 계정(k252a02)이 기본값이고, **SRC==DST 면 스크립트가 즉시 중단**한다.
- **⚠️ 다른 클러스터로 나갈 땐 이 스크립트를 쓸 수 없다**(홈 직접 read 전제). 데이터전송노드 `kbds-dm.kisti.re.kr`(FTP 21 / Aspera 33001) 경유.

---

## 3. 환경 함정 (겪고 기록한 것 — 재발 방지)

- **NVLink 없음**(PCIe A100, NCCL→SHM) → full-FT 375~660s/step. **전 단계 LoRA 필수**.
- **계산노드 컨테이너 불가**: CPU 노드 `max_user_namespaces=0` → Singularity 즉사. GPU 노드만 컨테이너 O. **데이터 변환은 conda `swift` env(pyarrow+PIL)로 CPU 잡** 실행(컨테이너 우회).
- **로그인노드 워치독**: 무거운 작업(대용량 이미지 추출 등) ~15분 무통보 kill. → 빌드는 CPU 잡으로. 로그는 항상 `python -u`.
- **HF 오프라인 플래그**: `00_common.sh` 가 `HF_HUB_OFFLINE=1` 설정 → 다운로드 시 런칭셸에서 **`unset HF_HUB_OFFLINE` + `--env HF_HUB_OFFLINE=0`**.
- **글로벌 vLLM 로그인노드 불가**(드라이버) → judge·학습·추론 전부 컴퓨트노드.
- **swift export 병합/eval 은 GPU 노드에서만**(컨테이너 필요). merge_lora 는 CPU 연산이지만 컨테이너가 CPU 노드서 안 도는 게 제약.
- **속도는 두 종류다.** `step_time` 은 학습 스텝만이고, 벽시계(`train_speed`)는 vLLM 롤아웃·sleep/wake·체크포인트를 포함해 **약 2배**다. 일정·예산 계산에는 **반드시 `train_speed` 쪽**을 쓸 것.
- **`num_generations` 는 계산이 아니라 데이터 노출을 깎는다.** 배치가 32 completion 고정이라 `num_gen` 을 올리면 프롬프트/step 이 줄어든다. 현재 **8 이 상한**.

### 🚨 apptainer 파손 → `ENV_MODE=loader` 우회 (2026-07-27~ · 현재도 유효)

**증상**: `singularity: error while loading shared libraries: libsubid.so.3` — 로그인·계산 노드 **모두**.
클러스터 공용 `/apps/application/apptainer/1.4.5` rpm 이 **GLIBC_2.28 요구**(호스트는 CentOS7/glibc 2.17)
+ `libsubid.so.3` 시스템 부재. **7/21 까지는 정상**(job 59191)이었고 원본 계정 로그에도 이 에러 없음
→ 그 사이 런타임이 교체/파손됨. **이미지 자체는 멀쩡**하므로 재빌드는 해결책이 아니다(빌드도 불가).

**우회**: sandbox 안에 Ubuntu22.04 의 완전한 **glibc 2.35 런타임**이 들어있다. 그 로더(`ld-linux`)로
sandbox python 을 직접 구동하면 호스트 glibc 를 우회해 동일 스택(torch/vllm/swift)이 그대로 돈다.
`00_common.sh` 의 `ENV_MODE` 기본값이 **`loader`** 이므로 **추가 조치 없이 기존 스크립트가 그대로 동작**한다.

```bash
./runc.sh -c "import torch; print(torch.__version__)"   # 단독 실행 확인
ENV_MODE=container bash scripts/...                      # apptainer 복구 후 원복
```
- 구현: `runc.sh`(로더 래퍼) + `bin/python`·`bin/python3`(torchrun 자식용 shim). 함정 5개는 `runc.sh` 주석 참조
  (sys.executable / PYTHONHOME / LD_LIBRARY_PATH 분리 / CUDA_HOME 실경로 / Triton 용 gcc 10.2.0).
- **검증**: job 72844 GRPO 2 step 완주 → 이후 8 GPU 본실행 1,047 step 완주(73924/73925)로 실전 검증됨.
- **apptainer 복구되면** `00_common.sh` 의 `ENV_MODE` 기본값을 `container` 로 되돌릴 것.

---

## 4. Stage-2 도메인 전문가 — 실행 절차 (현재 경로)

설계 근거 → [재설계](docs/stage2_redesign_2026.md) · [DeepSeek-V4 채택](docs/deepseek_v4_pipeline_adoption.md) · [서베이](docs/rlvr_survey_2026.md)

```bash
# 0) 확장셋 → 소스별 3분할 (이미지 경로 기준, 미분류 시 즉시 실패)
python3 scripts/split_stage2_by_source.py          # --dry / --verify

# 1) 배선 검증 — 유휴 debug-1gpu(40GB)에서 3 step. 8gpu 큐를 안 기다린다
bash scripts/probe_1gpu.sh
#    ⚠️ 여기서 step time·보상 크기는 읽지 않는다 (offload·1프롬프트라 통계가 아니다)

# 2) 8 GPU 스모크 — 5 step. 노드 3개 점유하니 큐 상황 보고
SMOKE=1 bash scripts/launch_domain_experts.sh

# 3) 본실행 3 arm (= 실제 제출 형태)
STEPS=1500 bash scripts/launch_domain_experts.sh
ARMS="mmk12 pmcvqa" bash scripts/launch_domain_experts.sh   # E1 빼고 ck-850 을 일반 교사로 쓰는 선택

# 4) 모니터
squeue -u $USER ; tail -f logs/grpo_adv_*.log
cat logs/verdict_<JID>.json          # 형식붕괴 감시자 판정 (WATCHDOG=1 잡)
bash scripts/watch_train.sh          # 라이브 대시보드 (tmux)
```

**재설계에서 바꾼 것** (붕괴 대응 — 전부 근거 있음):

| 바꾼 것 | 왜 |
|---|---|
| 도메인 3분할 | 혼합에서는 도메인별 길이·그룹 예산을 다르게 줄 수 없었다 |
| 정확도/형식 분리(`<answer>` 없으면 추론 꼬리에서 letter 복구) | 정확도가 형식에 물려 있어 붕괴가 자기증폭했다 |
| 단계형 `FormatThink` (`</think>` 까지면 0.5) | 붕괴 개시 절벽 −0.324 → −0.204 (37.1% 완화) |
| `recipe=stable` (`overlong_filter=false`, `scale_rewards=none`) | `overlong_filter` 는 붕괴를 막지 못하고 **회복을 막았다** |
| `num_generations` 4→8 | 붕괴 개시 시점 정확도 균일률 13.8%→54.2% |
| `lora_dropout=0` | RL 로그확률 계산 중 dropout 이 살아 π_rollout ≠ π_train 을 스스로 만들고 있었다 |
| `WATCHDOG=1` | 과거 로그 재생 결과 step 901 발화 · 오경보 0회 · 실제 발견보다 146 step(13.5h) 빠름 |

**판정 기준**: 각 전문가가 **자기 도메인 홀드아웃에서 init(`sft_mixed_merged`) 대비 상승**.
구 실행 기준선 = 전체 43.34% / deepvision 35.60 / 수학 48.25 / 의료 57.25 (n=1,772 전량).
⚠️ 과거 수치(v3 0.348 등)는 **구 홀드아웃** 기준이라 가로 비교 금지.

**감시 항목**: 스모크에서 `memory(GiB)` 52.7 → 65.2 → 74.1(80GB 의 93%). 본실행 초반에 평탄화 확인할 것.

---

## 5. 남은 일 (우선순위)

1. **전문가 3종 완주 감시** — `WATCHDOG` 판정 확인, 메모리 평탄화 확인. 붕괴 재발 시 즉시 중단(이미 자동).
2. **전문가별 홀드아웃 평가** → 도메인별 init 대비 이득 확인. `eval_paired.py` 로 McNemar.
3. **어댑터 통합** → Stage-3 init 만들기.
4. **Stage-3 본실행**: `bash scripts/launch_stage3.sh`. **계획서 핵심 산출물이고 미시작.** 잔여 예산 1,996 안에 넣어야 한다.
5. (옵션) v3 HealthBench Hard(n=1000, ~28 노드시간, gpu:2) — base 0.229/v2 0.224 대비.
6. (기록만) 구 DeepVision 홀드아웃 22% 오염 재구성 · 붕괴 개시 원인(재현 실패로 보류).

---

## 6. 파일 지도 (핵심만)

```
scripts/
  00_common.sh                     공통 경로/환경/run_py 래퍼  ← 이식 시 PROJ_DIR·ENV_MODE
  10_sft.slurm                     Stage-1 v3 SFT (기본값=sft_mixed)
  build_mixed_coldstart.py         v3 콜드스타트 데이터 빌드
  50_eval_v3.slurm / eval_v3_holdout.py   병합+홀드아웃 평가(strict format_think·층별)
  13_build_stage2_expanded.slurm   MMK12·ThinkLite 변환(CPU) — BUILD_PY 필요
  build_stage2_mix.py              확장셋 조립+홀드아웃(bytehash dedup)
  split_stage2_by_source.py        확장셋 → 소스별 3분할(+sha1 manifest)
  21_rlvr_grpo_adv.slurm           Stage-2 레시피(dapo|gspo|dr_grpo|stable) — 전 단계가 이걸 부른다
  launch_domain_experts.sh         🔴 현재 진입점. 전문가 3 arm 제출 (SMOKE / ARMS / DRY / STEPS)
  probe_1gpu.sh                    배선 검증 — 유휴 debug-1gpu 3 step
  watch_format_collapse.py         🚨 형식 붕괴 감시 → scancel. --simulate 로 과거 로그 재생
  eval_paired.py                   두 체크포인트 문항별 조인 → McNemar + 검출 하한
  plot_train_curves.py             학습 로그 → 6패널 곡선 + 구간 대조표
  30_medical_rl.slurm / launch_stage3.sh / judge_server.sh   Stage-3
  transfer_pull.sh                 계정 이관 데이터 pull (SRC 지정 필수)
  launch_stage2_expanded.sh        구 혼합 경로 — 붕괴로 중단, 재현용으로만 남긴다
configs/
  accuracy.py                      accuracy_mix + 단계형 format_think 보상
  medical_reward.py                Stage-3 RaR clinical_judge
docs/
  stage2_run73924_postmortem.md    🚨 붕괴 사후분석 rev.2 — 인수인계 시 먼저 읽을 것
  stage2_redesign_2026.md          현재 설계의 근거
  stage2_expansion_runbook.md      확장 재현 0~6단계
  medical_reward_spec.md · progress_log.md · worklog_*.md
(루트) 사후분석 일회성 스크립트 — 레포 루트에서 실행. 붕괴 분석 재현용
  holdout_matrix.py · holdout_by_source.py · train_side.py · classify900.py · show_collapse.py
  ⚠️ 입력이 logs/ 와 work/checkpoints/ 다 → transfer_pull 에서 WITH_LOGS 를 끄면 돌지 않는다
work/  (git 제외 — 재생성)          data·images·checkpoints·hf_cache·images/컨테이너
```

**막히면**: README 현황 → 이 문서 §1-붕괴 → §2(경로) → §3(함정) 순으로 확인.
