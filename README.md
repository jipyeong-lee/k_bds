# K-BDS 의료 멀티모달 교차추론 — 학습 파이프라인

계획서(`plan.hwp`) 기반, **ms-swift**로 4단계 파이프라인을 KISTI Slurm 클러스터에 맞춰 구성:
**① 콜드스타트 SFT → ② 범용 RLVR/GRPO → ③ 의료 특화 RL(RaR) → ④ 평가**.

> 이 문서는 **현황·요약·진입점**입니다. 상세 기록은 [문서 지도](#문서-지도) 참조.
> 계정 이식·인수인계 → [`HANDOFF.md`](HANDOFF.md)

---

## 현황 (2026-08-02)

**지금 위치**: **Stage-2 풀확장 본실행 중** (k252a02 이관 완료, MAX_STEPS=2,337 체인 job **73924~73927**) — **650/2,337 step (28%)** 통과.
**다음 임계경로**: **평가 검정력 보강**(n=300→1,200 + 짝지음, 검출 하한 8.0pp→2.8pp, ≈2~3 노드시간) → **step 1200 에서 사전기준 판정**(계속/중단) → **Stage-3 본실행**(계획서 핵심 산출물, 미시작 — 의료를 올리도록 설계된 유일한 단계).
**현재 중단 판단은 보류** — step 400 이후 상승이 검출되지 않았으나, 그것이 실제 포화인지 n=300 의 분해능 부족인지 구분되지 않는다.

| 단계 | 상태 | 핵심 결과 |
|---|---|---|
| **① 콜드스타트 SFT** | ✅ 완료 | v3 가 **형식 천장 완파**: 생성 `format_think` v2 **0.185**→v3 **0.909**(5배), 홀드아웃 acc **0.295→0.348**(+18%). `sft_mixed_merged` = Stage-2 init → [상세](docs/stage1_coldstart.md) |
| **② 범용 RLVR** | 🚀 **본실행 중 (28%)** · ❓중간평가 판정불가 | 확장셋 **74,787**(일반53/math20/의료26), init=v3·GDPO. **step 600 홀드아웃 = init 대비 +3.34pp(p=0.412)** → 추세 재측정 필요 → [중간점검](docs/stage2_run73924_progress.md) · [실험](docs/stage2_experiments.md) · [데이터](docs/stage2_data.md) · [실행현황](#stage-2-본실행-현황) |
| **③ 의료 RL (RaR)** | ⏳ 배선완료·대기 | 루브릭·judge(27B)·e2e 스모크 PASS(유닛 29/29) → [상세](docs/stage3_and_eval.md) |
| **④ 평가** | 🔄 기준선 확보 | HealthBench Hard(n=1000): base **0.229** / v2 콜드스타트 **0.224** — v3 미측정 → [상세](docs/stage3_and_eval.md) |

> ⚠️ **예산**: 신규 계정 5,000 노드시간 중 **Stage-2 본실행 ≈ 1,674(33%)** 집행, 나머지 67%는 **Stage-3·평가용 유보**. 구 계정 k252a01 은 83% 소진 후 이관.
>
> 🚨 **2026-08-02 정정 — "1 epoch" 표기 오류**: MAX_STEPS=2,337 을 1 epoch 으로 적어 왔으나 실제로는 **0.25 epoch** 이다.
> GRPO 에서 `per_device_train_batch_size` 는 프롬프트가 아니라 completion 을 세므로 **프롬프트/step = 32 ÷ num_generations(4) = 8**,
> **1 epoch = 74,787 ÷ 8 = 9,348 step**(로그의 `epoch=0.067` 이 이를 확증). 확장셋의 **약 75%는 미노출**로 남으며,
> 진짜 1 epoch 은 ≈6,694 노드시간 = **예산의 134%** 로 실행 불가. 집행 비용 자체는 step 기준이라 계획대로다.
> → [중간 점검 보고서](docs/stage2_run73924_progress.md) §3

> 🚨 **환경 (2026-07-27~)**: 클러스터 **apptainer 파손**(`libsubid.so.3` 부재 + GLIBC_2.28 요구 vs 호스트 2.17) → `singularity exec` 불가.
> **이미지는 정상**이라 재빌드는 무의미(빌드도 불가). **우회 = `ENV_MODE=loader` 가 기본값**이라 기존 스크립트가 그대로 동작
> (검증: 8GPU GRPO 완주, job 72832). apptainer 복구 시 `ENV_MODE=container` 로 원복. → [`HANDOFF.md`](HANDOFF.md) §3 · `runc.sh` 주석

### Stage-2 본실행 현황

**2026-08-03 09:00 KST 기준 · job 73925 (`gpu-8-002`) · 로그 `logs/grpo_adv_73925.log`**

| 항목 | 값 | 판정 |
|---|---|---|
| 진행 | **762 / 2,337 step (32.6%)** = 0.082 epoch | 정상 |
| 체인 | 73924 종료(TimeLimit, step 787) → **73925 가 checkpoint-750 에서 재개** · 73926/27 대기 | ✅ 인계 정상 |
| 속도 | **145 s/it** (최근 60 step 중앙값). 길이가 줄어 400~699 구간의 168 s/it 보다 빨라졌다 | ✅ |
| 잔여 추정 | 완주 ≈ 08-09 · **판정 시점(step 1200)까지 약 18h** | — |
| 안정성 | OOM · CUDA error · Traceback **0건** | ✅ 무결 |

> ⚠️ 재개 때마다 출력 디렉터리가 새로 생긴다 — 73924 `v0-20260731-094532`, 73925 **`v1-20260803-074645`**.
> 앞으로의 체크포인트는 v1 아래다. TimeLimit 컷은 마지막 저장(750) 이후 **37 step 을 버린다**(save_steps=50 구조상 정상).

**구간 대조 (1~100 step 평균 → 701~800 step 평균)** — `scripts/plot_train_curves.py` 출력

| 지표 | 초반 | 최근 | 변화 |
|---|---:|---:|---:|
| rewards/AccuracyMix | 0.4248 | 0.4533 | **+6.7%** |
| rewards/FormatThink | 0.9461 | 0.9600 | +1.5% |
| completions/mean_length | 1,127 | 1,163 | +3.2% (400~499 정점 1,660 에서 되돌아옴) |
| completions/clipped_ratio | 4.9% | 3.7% | **−25.4%** |
| kl | 0.0036 | 0.0313 | +765% (절대값은 작음) |
| entropy/mean | 0.5377 | 0.5311 | −1.2% (붕괴 없음) |

> ✅ **길이 인플레이션은 지나갔다.** 200~500 구간에서 길이 1,127→1,660자·클리핑 4.9→9.8%·형식준수 0.947→0.899 로 악화됐다가
> 700 step 대에 전부 되돌아왔다. **step 400/500/600 홀드아웃 평가는 하필 이 저점에서 찍힌 것**이다.
>
> 🔍 **최근 AccuracyMix 상승(+6.7%)의 대부분은 실력이 아니라 길이 구성이다.** 장문 completion 은 정답률이 훨씬 낮은데
> (39.4% vs 44.6%) 그 비중이 16.8% → 11.9% 로 줄면서 평균이 저절로 올랐다. 장문 제외 기준으로 남는 것은 +0.8pp(t=0.64, 미검출).
>
> ❌ **학습 로그로 "앞으로 오를까"를 답하려 했으나 실패했다.** `completions.jsonl` 24,000 문항(홀드아웃의 80배)으로도
> step 301~750 기울기는 deepvision **−0.05pp/100step, 95% 구간 −14.4 ~ +12.9pp** — 세 소스 모두 0 을 크게 포함한다.
> 수준을 재는 것과 기울기를 재는 것은 다른 문제다. **판정은 step 1200 홀드아웃(n=1,200 + 짝지음)이 유일한 근거다.**
> → [보고서 §6-c](docs/stage2_run73924_progress.md) · 재현 `python3 scripts/train_source_trend.py --until 750`

![소스별 학습 정확도 추세](docs/assets/stage2_source_trend.png)

**체인 소진 예측**: 73924 는 벽시계 한계(TimeLimit 2-22:00:00, 08-03 07:39 종료)에서 **~778 step** 도달 후 resume 인계.
잔여 3잡이 각 ~782 step 을 담당해 **73926 중 2,337 step 도달** 예상 → **완주 ≈ 08-09**, 73927 은 여유분.

**중간 홀드아웃 추세 (job 74060·74062·74063, n=300 층화)** — 확장 홀드아웃, greedy, 전 모델 동일 슬라이스

| | base | **init**(RL 0%) | step 400 | step 500 | step 600 |
|---|---:|---:|---:|---:|---:|
| **전체** | 0.2500 | **0.4533** | **0.4900** | 0.4800 | 0.4867 |
| deepvision / mmk12 / **pmcvqa** | 0.12 / 0.31 / 0.32 | 0.39 / 0.40 / **0.57** | 0.43 / 0.48 / 0.56 | 0.37 / 0.50 / 0.57 | 0.45 / 0.48 / **0.53** |

![Stage-2 홀드아웃 정확도 추세](docs/assets/stage2_holdout_trend.png)

> ⚠️ **step 400 이후는 판별 불가**: init → 400 은 +3.67pp 이지만 **400·500·600 은 폭 1.0pp** 로 서로 구분되지 않는다.
> 다만 이는 **포화의 증거가 아니라 분해능 부족**이다 — n=300 의 검출 하한은 **8.0pp** 로, 관측 폭 1.0pp 는
> "차이 없음"이 아니라 **"이 표본으로는 볼 수 없음"** 이다.
> 학습 로그(24,000 문항)로 이 공백을 메우려 했으나 **거기서도 기울기는 판별되지 않았다**(§6-c).
> → **n=1,200 + 짝지음으로 하한을 2.8pp 로 낮추고 step 1200 에서 판정**한다. 그때까지 학습은 계속한다.
> 대조적으로 base → init 은 **+20.33pp, p<0.001** 로 명확히 유의 → 평가 설계 자체는 실효 효과를 잡아낸다.
> 목표 도메인인 **의료(pmcvqa)는 상승 신호 없음**(0.57→0.56→0.57→0.53) — 단 소스별 n=100 이라 ±9.8pp 이고,
> 학습 로그에서도 추세 미검출이다(다만 CI ±23pp 로 무의미). n=400 재측정에서 갈린다.
> ⚠️ 과거 수치(v3 0.348 등)는 **구 홀드아웃** 기준이라 가로 비교 금지.

📊 **학습 곡선·전체 분석 → [`docs/stage2_run73924_progress.md`](docs/stage2_run73924_progress.md)**

![Stage-2 확장셋 GDPO 학습 곡선](docs/assets/stage2_expanded_73924_curves.png)

---

## 파이프라인 4단계

| 단계 | 목적 | 데이터 | 방법 |
|---|---|---|---|
| **①** 콜드스타트 SFT | `<think>/<answer>` 형식 주입 + 의료 추론 시드 | v3: OpenMedReason + VisualWebInstruct + VLAA 혼합 | LoRA SFT |
| **②** 범용 RLVR | 검증가능 정답으로 추론 강화 | DeepVision 40K + MMK12 + PMC-VQA(의료) = **74,787** | GRPO **GDPO** · init=v3 |
| **③** 의료 특화 RL | 개방형 의료 VQA 추론 | medix-rl-data 51K | GRPO + RaR 루브릭 보상 |
| **④** 평가 | base 대비 성능 정량화 | 층화 홀드아웃 / HealthBench | vLLM 추론·채점 |

**공통 제약**: NVLink 없음 → **전 단계 LoRA-DDP** · glibc 2.17 → **컨테이너 스택**(현재 loader 우회) · 로그인노드 vLLM 불가 → **모든 GPU 작업은 컴퓨트노드**.
→ [기술 레퍼런스](docs/tech_reference.md)

---

## 빠른 실행

```bash
# Stage-2 풀확장 (현재 진행중인 경로)
SMOKE=1 bash scripts/launch_stage2_expanded.sh    # 배선 스모크(max_steps 5) — 먼저 권장
bash scripts/launch_stage2_expanded_epoch.sh      # 2,337 step(=0.25 epoch) 체인 4잡, ~209h
#   기본값(=검증값, 그대로 둘 것): RECIPE=dr_grpo · NUM_GEN=4 · TEMPERATURE=0.9 · BETA=0.04
#   A/B override 후보:            NUM_GEN=8 · TEMPERATURE=1.0 · BETA=0.01  ← 바꾸면 과거 A/B 와 clean 비교가 깨짐

# 단계별 단독 제출 (의존성 체이닝)
JID1=$(sbatch --parsable scripts/10_sft.slurm)                                   # Stage-1
JID2=$(sbatch --parsable --dependency=afterok:$JID1 scripts/20_rlvr_grpo.slurm)  # Stage-2
JID3=$(sbatch --parsable --dependency=afterok:$JID2 scripts/30_medical_rl.slurm) # Stage-3
sbatch          --dependency=afterok:$JID3 scripts/40_eval.slurm                 # 평가

# 모니터
squeue -u $USER ; tail -f logs/grpo_adv_*.log
```
- 재현 절차 상세 → [`docs/stage2_expansion_runbook.md`](docs/stage2_expansion_runbook.md)
- ⚠️ 체인 중단 시 **4개 job 전부 `scancel`** (하나만 취소하면 다음 잡이 이어받음)
- ⚠️ 학습량은 **에포크로 늘리지 말고 홀드아웃 포화로 조기중단** → [`docs/rlvr_hparams_external.md`](docs/rlvr_hparams_external.md)
- ⚠️ `MAX_STEPS` 는 **step 단위**로만 해석할 것. 1 epoch = 9,348 step 이며 예산상 도달 불가 → [정정 근거](docs/stage2_run73924_progress.md#3-epoch-커버리지--계획-전제가-4배-틀렸다)

---

## 문서 지도

| 문서 | 내용 |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | **인수인계 단일 문서** — 계정 이식·환경 함정·실행 절차 |
| [`docs/stage1_coldstart.md`](docs/stage1_coldstart.md) | Stage-1 상세 — v2 형식 천장 진단, v3 설계·학습곡선·홀드아웃 평가, ablation |
| [`docs/stage2_experiments.md`](docs/stage2_experiments.md) | Stage-2 실험 — plateau 진단, GRPO 계열 5종 clean A/B, 벤치마크 |
| [`docs/stage2_overview_for_slides.md`](docs/stage2_overview_for_slides.md) | 📊 **발표용 자립 요약** — 방법론 계보·데이터셋 선별·학습 세팅·진행 경과·홀드아웃 추세를 한 문서로 (절=슬라이드 1장) |
| [`docs/stage2_run73924_progress.md`](docs/stage2_run73924_progress.md) | **본실행 중간 점검(step 650)** — 학습 곡선 6패널, 정확도 정지·길이 인플레이션 진단, epoch 커버리지 정정, 홀드아웃 추세 |
| [`docs/stage2_data.md`](docs/stage2_data.md) | Stage-2 데이터 — 소스 스크리닝(실측)·혼합비율·빌드 파이프라인 |
| [`docs/stage2_expansion_runbook.md`](docs/stage2_expansion_runbook.md) | Stage-2 풀확장 재현 0~6단계 |
| [`docs/rlvr_hparams_external.md`](docs/rlvr_hparams_external.md) | RLVR 하이퍼파라미터 — 2026 리포트 외부 관행 대조·에포크 정책 |
| [`docs/stage3_and_eval.md`](docs/stage3_and_eval.md) | Stage-3 의료 RL(RaR 루브릭·judge) + HealthBench 추적 |
| [`docs/medical_reward_spec.md`](docs/medical_reward_spec.md) | 의료 보상 스펙 |
| [`docs/tech_reference.md`](docs/tech_reference.md) | 환경·베이스모델·LoRA·보상 설계·논문 레퍼런스 |
| [`docs/ops_data.md`](docs/ops_data.md) | 자원·운영 정책, 데이터 소스, **홀드아웃 분리(누수 차단)** |
| [`docs/progress_log.md`](docs/progress_log.md) | 진행 이력·핵심 의사결정·날짜별 기록·TODO |
| [`docs/project_status_2026-07-05.md`](docs/project_status_2026-07-05.md) | 문제정의·실험결과·해결방안·목표·기한 4축 |
| `docs/worklog_*.md` | 일별 상세 로그 |

---

## 파일 지도 (핵심)

```
scripts/
  00_common.sh                      공통 경로/환경/run_py 래퍼 ← 이식 시 PROJ_DIR·ENV_MODE
  10_sft.slurm                      Stage-1 v3 SFT
  20_rlvr_grpo.slurm                Stage-2 GRPO (기본=v3 init + 확장셋)
  21_rlvr_grpo_adv.slurm            Stage-2 검증된 레시피(dr_grpo/GDPO · dynamic_sample)
  launch_stage2_expanded.sh         Stage-2 표준 진입점(단발)
  launch_stage2_expanded_epoch.sh   Stage-2 스텝 체인(resume, MAX_STEPS=2,337 = 0.25 epoch)
  build_stage2_mix.py               확장셋 조립(bytehash dedup)
  plot_train_curves.py              학습 로그 → 6패널 곡선 + 구간 대조표(체인 로그 병합 지원)
  plot_eval_trend.py                평가 결과 jsonl → step별 홀드아웃 정확도 추세(95% CI)
  eval_midtrain.slurm               중간 홀드아웃 평가(EVAL_STAGES 로 대상 선택, 병렬 제출 가능)
  30_medical_rl.slurm · launch_stage3.sh · judge_server.sh    Stage-3
  50_eval_v3.slurm · eval_v3_holdout.py                       평가
configs/   accuracy.py(Stage-2 보상) · medical_reward.py(Stage-3 RaR)
runc.sh · bin/python                apptainer 우회 런타임(ENV_MODE=loader)
work/      (git 제외) data · images · checkpoints · hf_cache
```

---

## 과제 종료 의무 (가이드 §7)
- 종료 후 **1주 내** 데이터 다운로드(이후 차단·삭제) · **1개월 내** 결과보고서 + 산출물 기탁(marketplace.kbds.re.kr) · **2년 내** 사사표기 논문.
- 사사: *"이 논문은 K-BDS로부터 컴퓨팅 자원과 기술지원을 받아 수행된 연구성과임"* / *"This work was supported by the Korea Bio Data Station(K-BDS) with computing resources including technical support"*
