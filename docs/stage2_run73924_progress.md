# Stage-2 확장셋 GDPO 본실행 — job 73924 중간 점검

**작성 2026-08-02 20:13 KST · 대상 job 73924(`gpu-8-002`) · 로그 `logs/grpo_adv_73924.log` · step 650 시점**

> 재생성: `./bin/python scripts/plot_train_curves.py logs/grpo_adv_73924.log -o docs/assets/stage2_expanded_73924_curves.png --max-steps 2337`

체인 job **73924~73927** 중 첫 잡이 진행 중이다. 이 문서는 ① 학습이 건강하게 도는지, ② 실제로 무엇이 학습되고 있는지,
③ 자원이 계획대로 쓰이는지 세 가지를 실측으로 점검한다.
**결론부터: 인프라는 무결하나 정확도 보상이 650 step 동안 정지해 있고, 계획의 "1 epoch" 전제가 4배 틀렸다.
중간 홀드아웃 평가(§6)는 init 대비 +3.34pp 로 유의하지 않아 계속/중단 어느 쪽도 확정하지 못한다 — 추세 재측정이 필요하다.**

---

## 1. 요약

| 항목 | 실측 | 판정 |
|---|---|---|
| 진행 | 650 / 2,337 step (27.8%) | 정상 |
| 안정성 | OOM · CUDA error · Traceback **0건**, mem 90.9 GiB | ✅ 무결 |
| **정확도 보상** | AccuracyMix 0.425 → 0.431 (**+1.3%**) | 🚨 **정지** |
| 형식 보상 | FormatThink 0.946 → 0.929 (−1.8%) | 이미 포화(천장) |
| 길이 | mean_length 1,127 → 1,444 tokens (**+28.1%**) | ⚠️ 팽창 |
| 클리핑 | clipped_ratio 4.9% → 7.0% (**+41.4%**) | ⚠️ 증가 |
| KL | 0.0036 → 0.0262 (**+626%**) | 정책 이탈 진행(절대값은 아직 작음) |
| **홀드아웃 (§6)** | init 0.4533 → trained(600) 0.4867 = **+3.34pp**, p=0.412 | ❓ **판정 불가** |
| └ 의료(pmcvqa) | 0.57 → 0.53 (−4pp, 유의하지 않음) | ⚠️ 유일한 하락 |
| **epoch 커버리지** | 2,337 step = **0.25 epoch** (1 epoch = 9,348) | 🚨 **계획 오류** |
| 오버헤드 | step_time 합 29.0h vs 실경과 58.2h → **50%가 step 밖** | ⚠️ 개선 여지 |

---

## 2. 학습 곡선

![Stage-2 확장셋 GDPO 학습 곡선 — job 73924](assets/stage2_expanded_73924_curves.png)

*옅은 선 = step별 원본(logging_steps=1) · 굵은 선 = 25-step 이동평균 · 우측 수치 = 이동평균 종점.*

### 구간 대조 (1~100 step 평균 vs 551~650 step 평균)

`scripts/plot_train_curves.py` 가 이 표를 그대로 출력한다 — 갱신 시 재실행할 것.

| 지표 | 1~100 | 551~650 | 변화 |
|---|---:|---:|---:|
| reward (총합) | 0.6019 | 0.5983 | −0.6% |
| rewards/AccuracyMix/mean | 0.4248 | 0.4305 | **+1.3%** |
| rewards/FormatThink/mean | 0.9461 | 0.9291 | −1.8% |
| rewards/SoftOverlong/mean | −0.0606 | −0.0900 | −48.5% (페널티 증가) |
| kl | 0.0036 | 0.0262 | +625.7% |
| entropy/mean | 0.5377 | 0.5280 | −1.8% |
| completions/mean_length | 1,127 | 1,444 | +28.1% |
| completions/clipped_ratio | 0.0494 | 0.0698 | +41.4% |
| reward_std | 0.4951 | 0.4843 | −2.2% |
| frac_reward_zero_std | 0.0044 | 0.0187 | +328.6% |
| step_time | 146.1s | 164.0s | +12.2% |

### 읽기

- **정확도가 학습되지 않고 있다.** 650 step 동안 AccuracyMix 는 0.425 → 0.431 로, 노이즈 폭(step별 표준편차 ≈ 0.50) 안에서 사실상 정지다.
  총 reward 가 −0.6% 인 것은 형식 보상 소폭 하락과 SoftOverlong 페널티 증가가 겹친 결과이며, **정확도 기여분은 사실상 0** 이다.
- **대신 길이가 팽창하고 있다.** 평균 completion 이 +28.1%, 6,144 토큰 클리핑이 +41.4%, SoftOverlong 페널티가 −48.5%.
  정확도 이득 없이 출력만 길어지는 이 조합은 RLVR 의 전형적인 **길이 인플레이션** 패턴이다.
- **엔트로피는 붕괴하지 않았다** (0.538 → 0.528). 다양성 붕괴(diversity collapse) 징후는 아직 없다 — `docs/rlvr_hparams_external.md` 의 경고 구간에는 미도달.
- **KL 은 626% 늘었지만 절대값 0.026 으로 작다.** β=0.04 제약이 살아 있고 clip_ratio 도 0.0009 수준이라 정책이 폭주하는 상태는 아니다.
  다만 250~400 step 구간에서 가파르게 상승했고 이 구간이 길이 팽창 시작점과 겹친다.
- **`frac_reward_zero_std` 가 0.0044 → 0.0187 로 4.2배** 늘었다. 그룹 내 모든 롤아웃이 같은 보상을 받아 **advantage 가 0 이 되는 배치**가 늘고 있다는 뜻으로,
  학습 신호가 마르는 방향이다. `dynamic_sample`(max_resample 3)이 아직 흡수하고 있으나 추세는 주시할 것.
- `All completions are overlong and truncated` 경고 **203회**(650 step 대비 31%). 해당 step 은 KL 이 NaN 으로 빠진다. 학습은 계속되지만 그 구간의 KL 제약이 사실상 무효다.

---

## 3. epoch 커버리지 — 계획 전제가 4배 틀렸다

`scripts/launch_stage2_expanded_epoch.sh` 는 1 epoch 을 이렇게 산정했다.

```
1 epoch = 74,787 / 32(step당 프롬프트: 1 × accum4 × 8gpu) ≈ 2,337 step
```

**이 32 는 프롬프트 수가 아니라 completion 수다.** GRPO 에서 `per_device_train_batch_size` 는 completion 을 세며,
`generation_batch_size` 가 `None` 이므로 TRL/ms-swift 기본식이 적용된다.

```
generation_batch_size = pdtbs(1) × world_size(8) × steps_per_generation(=grad_accum 4) = 32 completions
프롬프트/step        = 32 ÷ num_generations(4)                                        =  8 prompts
1 epoch              = 74,787 ÷ 8                                                     = 9,348 step
```

**로그가 이를 확증한다**: step 627 × 8 = 5,016 행, 5,016 / 74,787 = **0.06706** — 로그의 `epoch: 0.06707` 과 일치.
(공식이 맞다면 `epoch` 이 0.268 로 찍혀야 한다.)

### 결과

| | 계획이 믿은 값 | 실제 |
|---|---|---|
| MAX_STEPS=2,337 의 의미 | 1 epoch (100%) | **0.25 epoch** |
| 현재 630 step | 0.27 epoch | **0.067 epoch** |
| 확장셋 74,787건 중 노출 | 전량 | **약 25%** |
| 진짜 1 epoch(9,348 step) 비용 | — | ≈ 837h · **6,694 노드시간 = 예산 5,000의 134%** |

**예산은 안전하다.** `max_steps=2337` 이 상한이라 `num_train_epochs=3.0` 설정과 무관하게 2,337 에서 멈추며,
2,337 step × 322 s/it × 8 GPU ≈ **1,674 노드시간**으로 계획서의 1,719(34%) 와 사실상 같다. 지금 도는 job 과 비용은 계획대로다.

**틀린 것은 라벨과 그로부터 나온 판단 근거다.**

1. 확장셋의 **75%는 한 번도 학습에 노출되지 않는다.** 이번 확장의 핵심이던 MMK12(math 20%)·PMC-VQA(의료 26%) 추가분 상당량이 미사용으로 남는다.
2. 따라서 중간 평가에서 홀드아웃이 포화로 보여도 **방법의 한계인지 데이터 미노출 탓인지 구분되지 않는다.** "포화 시 조기중단" 정책의 전제가 흔들린다.
3. "2 epoch 은 69% 라 미채택" 이라는 서술은 실제로는 "0.5 epoch" 이야기다. 진짜 1 epoch 은 예산의 134% 로 **애초에 실행 불가능한 계획**이었다.

> 다만 §2 의 실측을 보면 데이터 미노출이 정체의 주원인이라고 단정할 수 없다. 650 step 동안 본 5,200 개 프롬프트만으로도
> 정확도 보상이 전혀 안 오르는 상태라, **데이터 양보다 보상 신호·길이 예산 쪽 문제일 가능성이 높다.**

---

## 4. 자원 — 오버헤드 50%

| | 값 |
|---|---|
| step_time 합계 (650 step) | 29.0h |
| 실제 경과 | 58.2h |
| **step 밖 오버헤드** | **29.2h (50.2%)** |
| 실효 속도 | 322 s/it (step_time 161s 의 2.0배) |

`--use_vllm true --vllm_mode colocate --sleep_level 1` 구성에서 롤아웃 생성·sleep/wake 전환,
`--dynamic_sample true --max_resample_times 3` 재샘플링, `save_steps 50` 체크포인트가 여기에 들어간다.
GRPO colocate 의 구조적 비용이라 전부 제거할 수는 없으나, **절반이 학습 밖에 쓰이고 있다는 사실은 기록해 둔다.**

---

## 5. 체인 소진 예측

- 73924 는 TimeLimit `2-22:00:00`(종료 2026-08-03 07:39)에서 **~778 step** 도달 후 resume 인계
- 잔여 3잡이 각 70h ≈ 782 step 담당 → **73926 중 2,337 step 도달**, 완주 ≈ **2026-08-09** (잡 간 큐 대기 별도)
- 73927 은 여유분 (MAX_STEPS 도달 시 자동 no-op)

---

## 6. 중간 홀드아웃 평가 — job 74060 (2026-08-02 완료)

`scripts/eval_midtrain.slurm` · `debug-1gpu` · 55분 · ExitCode 0.
`stage2_expanded_holdout.jsonl`(1,772)에서 `_source` 별 **100건씩 층화 추출한 300건**을
base / init(RL 0%) / trained(step 600) 세 모델에 **동일 슬라이스·동일 조건**(greedy, `temperature=0.0`,
같은 노드·vLLM·시스템 프롬프트)으로 채점.

> ⚠️ 이 평가를 돌리려면 평가 경로를 loader 로 포팅해야 했다([a22d16c]) — 평가 스크립트 9종이 전부
> apptainer 파손(07-27) **이전** 작성이라 `singularity exec` 직접 호출로 실행 불가 상태였다.
> 본실행 2.5일째까지 중간 평가가 한 번도 못 돈 이유다. 표본 추출 버그(앞에서 N줄 자르기 → 전부
> deepvision)도 함께 고쳤다([cf3ee48]).

| 지표 | base | init (RL 0%) | trained (step 600) |
|---|---:|---:|---:|
| **accuracy (n=300)** | 0.2500 | **0.4533** | **0.4867** |
| format(`<answer>`) | 0.333 | 0.960 | 0.937 |
| mean_chars | 5,607 | 1,774 | 1,906 |
| errors | 0/300 | 0/300 | 0/300 |
| deepvision (n=100) | 0.12 | 0.39 | 0.45 |
| mmk12 · math (n=100) | 0.31 | 0.40 | 0.48 |
| **pmcvqa · 의료 (n=100)** | 0.32 | **0.57** | **0.53** |

층별(`_stratum`): numeric 0.538→0.577→**0.769** · vl 0.143→0.464→0.518 · math 0.091→0.296→0.364 ·
letter 0.320→0.570→0.530 · symbolic 0.026→0.132→**0.079** · other 0.200→0.500→0.500

### 판정 — 사전에 정한 "애매" 구간

**init → trained = +3.34pp.** 경계(±3pp)를 아슬아슬하게 넘겼으나 **통계적으로 0과 구분되지 않는다.**

| 비교 | 차이 | 95% CI | p | 판정 |
|---|---:|---|---:|---|
| **trained − init** (전체) | **+3.34pp** | [−4.64, +11.32]pp | 0.412 | 유의하지 않음 |
| deepvision | +6.00pp | [−7.66, +19.66]pp | 0.389 | 유의하지 않음 |
| mmk12 (math) | +8.00pp | [−5.71, +21.71]pp | 0.253 | 유의하지 않음 |
| pmcvqa (의료) | −4.00pp | [−17.78, +9.78]pp | 0.569 | 유의하지 않음 |
| *(참고)* init − base | +20.33pp | [+12.86, +27.80]pp | **<0.001** | **유의** |

같은 표본을 쓴 대응(paired) 비교로 보정해도 상관 r=0.7 가정에서 p=0.134 로 유의하지 않다.
반면 base→init 은 명확히 유의하다 — **이 평가 설계는 실재하는 효과를 잡아낸다.** 콜드스타트 SFT 는
확실히 작동했고, RL 600 step 은 그만한 신호를 내지 못하고 있다.

### 읽기

- **의료가 유일한 하락 방향이다.** pmcvqa 0.57 → 0.53. 유의하지 않은 변동이나 **프로젝트 목표 도메인**이
  유일하게 내려갔다는 점은 가볍게 넘길 신호가 아니다. 반대로 numeric(mmk12) 은 0.577 → 0.769 로 최대 상승.
  잠정적으로 **일반·수학은 오르고 의료는 정체~하락**이라는 그림이다.
- **길이 인플레이션이 홀드아웃에서도 재현된다.** mean_chars 1,774 → 1,906(+7.4%), format 0.960 → 0.937.
  §2 의 학습측 관측(길이 +28.1%, FormatThink −1.8%)과 같은 방향이다.
- **base 의 낮은 점수에는 형식 미준수가 섞여 있다**(format 0.333). `accuracy_mix` 가 `<answer>` 를 파싱하므로
  형식을 못 지키면 정답이어도 감점된다. 다만 판단에 쓰는 init vs trained 는 둘 다 형식이 잡혀 있어 무관하다.

### 검정력 — n=300 으로는 애초에 못 잡는다

| 잡으려는 차이 | 필요한 모델당 n (검정력 80%, 양측 5%) |
|---|---:|
| 8pp | ≈ 613 |
| 5pp | ≈ 1,566 |
| 3pp | ≈ 4,339 |

n=300 은 **8pp 이상만** 검출한다. 즉 +3.34pp 는 "효과가 없다"가 아니라 **"이 표본으로는 알 수 없다"** 이다.
홀드아웃 전량(1,772)을 쓰면 모델당 약 590 건까지 올릴 수 있어 8pp 는 확실히, 5pp 는 부분적으로 잡힌다.

<details><summary>원시 결과 (logs/ 는 gitignore 라 여기 보존)</summary>

```json
{"tag": "base", "n": 300, "accuracy": 0.25, "format": 0.333, "mean_chars": 5607.0, "errors": 0, "per_stratum": {"math": 0.0909, "vl": 0.1429, "symbolic": 0.0263, "numeric": 0.5385, "other": 0.2, "letter": 0.32}, "per_source": {"deepvision": 0.12, "mmk12": 0.31, "pmcvqa": 0.32}}
{"tag": "init", "n": 300, "accuracy": 0.4533, "format": 0.96, "mean_chars": 1774.0, "errors": 0, "per_stratum": {"math": 0.2955, "vl": 0.4643, "symbolic": 0.1316, "numeric": 0.5769, "other": 0.5, "letter": 0.57}, "per_source": {"deepvision": 0.39, "mmk12": 0.4, "pmcvqa": 0.57}}
{"tag": "trained", "n": 300, "accuracy": 0.4867, "format": 0.937, "mean_chars": 1906.0, "errors": 0, "per_stratum": {"math": 0.3636, "vl": 0.5179, "symbolic": 0.0789, "numeric": 0.7692, "other": 0.5, "letter": 0.53}, "per_source": {"deepvision": 0.45, "mmk12": 0.48, "pmcvqa": 0.53}}
```

재현: `sbatch --partition=debug-1gpu --export=ALL,MID_CKPT=$PWD/work/checkpoints/_mideval_snap_step600,EVAL_DATA=work/data/stage2_expanded_holdout.jsonl,EVAL_N=300,EVAL_CONC=16 scripts/eval_midtrain.slurm`

</details>

> ⚠️ **과거 수치와 가로 비교 금지**: 문서에 남은 v3 0.348 · dr_grpo 0.380 · GDPO 0.390 · GSPO 0.290 은
> **구 홀드아웃**(`deepvision_holdout.jsonl`, 972)에서 잰 값이다. 위 표는 확장 홀드아웃 층화 표본이라
> 모집단이 다르다. 굳이 대응시키면 init 의 deepvision 층 0.39 가 과거 v3 0.348 과 같은 계열이다.

---

## 7. 권고

| # | 조치 | 근거 | 우선도 |
|---|---|---|---|
| 1 | **추세 평가** — init / ckpt-400 / 500 / 600 을 모델당 n≈590(홀드아웃 전량)으로 재측정 | §6 — n=300 은 8pp 미만을 못 잡는다. 점 하나가 아니라 **기울기**가 있어야 포화인지 상승 중인지 갈린다. 스냅샷 `_mideval_snap_step{400,500,600}` 확보됨 | **높음** |
| 2 | **길이 예산 재검토** — `max_completion_length` 6,144 대비 클리핑 7.0%·경고 203회 | 학습(길이 +28.1%)과 홀드아웃(mean_chars +7.4%, format 0.960→0.937) 양쪽에서 인플레이션 재현. `SoftOverlong` 가중치(0.2)·`soft_cache_length` 재조정 후보 | **높음** |
| 3 | **의료 하락 추적** — pmcvqa 0.57 → 0.53 | §6 — 목표 도메인이 유일한 하락 방향. 추세 평가에서 재현되면 보상·데이터 혼합비 재검토 | **높음** |
| 4 | 나머지 **평가 스크립트 8종 loader 포팅** | §6 — 전부 `singularity exec` 직접 호출로 깨져 있다. `run_serve` 한 줄 교체면 복구 | 중 |
| 5 | 데이터 양보다 **보상 설계 재검토** | §3 말미 — 5,200 프롬프트에서도 정확도 신호가 안 잡힌다 | 중 |
| 6 | 오버헤드 50% 프로파일링 (생성 vs 재샘플 vs 체크포인트 분해) | §4 — 절반을 회수하면 동일 예산으로 2배 학습 | 중 |

*완료: 중간 홀드아웃 평가 실행(§6) · 문서 "1 epoch" 오류 전면 정정(§3) · 평가 경로 loader 복구*

**계속/중단은 ①의 추세가 나온 뒤에 판단한다.** 현 시점 근거로는 어느 쪽도 확정할 수 없다 —
step 600 의 +3.34pp 는 유의하지 않지만(p=0.412) 음수도 아니다. 인프라는 무결하고 예산도 계획 내이므로
지금 당장 중단할 이유는 없으나, **남은 1,687 step ≈ 1,210 노드시간**(예산 5,000 의 24%)을 계속 쓸 근거도
아직 없다. Stage-3(계획서 핵심 산출물, ~500 노드시간)가 미시작이라는 점이 이 결정의 기회비용이다.

---

## 관련 문서

- [`README.md`](../README.md) — 현황 요약
- [`stage2_experiments.md`](stage2_experiments.md) — Stage-2 실험 이력·A/B
- [`stage2_expansion_runbook.md`](stage2_expansion_runbook.md) — 재현 절차
- [`rlvr_hparams_external.md`](rlvr_hparams_external.md) — 하이퍼파라미터 외부 관행 대조
- [`ops_data.md`](ops_data.md) — 자원·예산

## 재생성

곡선·구간 대조표 모두 `scripts/plot_train_curves.py` 하나로 다시 만든다.

```bash
# 곡선 + 구간 대조표 (matplotlib 은 loader python 에만 있다)
./bin/python scripts/plot_train_curves.py logs/grpo_adv_73924.log \
    -o docs/assets/stage2_expanded_73924_curves.png --max-steps 2337

# 체인 잡을 이어서 보기 (step 기준 병합, resume 구간은 나중 파일이 이김)
./bin/python scripts/plot_train_curves.py logs/grpo_adv_739*.log \
    -o docs/assets/stage2_expanded_chain_curves.png --max-steps 2337

# 수치만 (호스트 python3 로도 동작 — matplotlib 불필요)
python3 scripts/plot_train_curves.py logs/grpo_adv_73924.log --no-plot --csv logs/train_metrics.csv
```

계산노드처럼 CJK 폰트가 없는 환경에서는 레이블이 자동으로 영문으로 떨어진다(수치·레이아웃은 동일).
