# Stage-2 재실행 실험 설계

> 작성 2026-08-13 · 근거 = [2026년 모델 서베이](rlvr_survey_2026.md) + [붕괴 사후분석 rev.2](stage2_run73924_postmortem.md)
> 대상 = job 73924/73925 붕괴 이후의 Stage-2 재실행 설정
>
> **설계 원칙**: 서베이에서 가져온 것과 우리 로그에서 잰 것을 섞지 않는다.
> 각 항목에 **근거 출처**와 **실측 효과**를 따로 적는다. 실측이 없는 항목은 그렇다고 적는다.

---

## 0. 재현 실험이 바꾼 전제

먼저 짚어야 할 것 — **개시 원인을 찾는 일은 끝났다.**

checkpoint-800 에서 설정 28개를 전부 동일하게 맞춰 재실행한 결과(job 74583, step 801→927),
**붕괴가 재현되지 않았다.** 전조 램프(KL 0.030→0.121)조차 재현되지 않았다. 재개가
옵티마이저·LR·데이터로더 위치까지 복원했으므로 달랐던 자유변수는 vLLM 롤아웃 샘플링뿐이다.

> **따라서 "step 900 에 터지게 되어 있었다"는 설명은 틀렸다.** 개시는 확률적이었다.
> 설계 목표가 바뀐다 — **개시를 막는 것이 아니라, 개시가 절벽으로 이어지지 않게 만드는 것.**

이 문서의 변경안은 전부 이 목표에 정렬되어 있다.

---

## 1. 요약

| | 변경 | 근거 | 실측 효과 | 비용 |
|---|---|---|---|---|
| **A** | 정확도 보상을 형식에서 분리 ✅완료 | 자체 실측 + 서베이 ① | 개시 낙폭 −0.324 → **−0.254** | 0 |
| **B** | **형식 보상을 이진→단계형** | 자체 실측 + [GLM-4.1V] | 개시 낙폭 −0.254 → **−0.195** | 0 |
| **C** | `overlong_filter` → **false** | 자체 실측(§4-4) | 회복 차단 요인 제거 | 0 |
| **D** | `scale_rewards` gdpo → **none** | [EXAONE 4.5](https://arxiv.org/abs/2604.08644) | 없음(위생) | 0 |
| **E** | `max_steps` 를 실제 정지 step 으로 | 설계 오류 수정 | 없음 | 0 |
| **F** | **rollout logprob 불일치 계측** | [IcePop](https://ant-ling.medium.com/ring-flash-2-0-4bf5b62204f5) | 미측정 — **재는 게 목적** | 낮음 |
| **G** | 감시자 상시 가동 | 자체 검증 | 실제 발견보다 146 step 빠름 | 0 |
| **H** | 길이 하드 예산 | [Kimi K3](https://arxiv.org/abs/2607.24653) | 미측정 | 코드 |
| **I** | `loss_type=sapo` | [Qwen3-VL/SAPO](https://arxiv.org/html/2511.20347v1) | ⚠️ **보류 — 트레이드오프** | 0 |

**A+B 만으로 개시 시점 총보상 낙폭이 −0.324 → −0.195 로 40% 줄어든다.** 둘 다 GPU 비용 0 이다.

---

## 2. 좋은 소식 — ms-swift 4.13 이 이미 지원한다

서베이에서 찾은 2026년 기법 대부분이 **이미 플래그로 들어와 있다.** 커스텀 코드가 필요 없다.
(설치본 `work/images/ms-swift-413-sandbox/.../swift/rlhf_trainers/args_mixin.py` 직접 확인)

| 서베이 발견 | ms-swift 인자 | 상태 |
|---|---|---|
| SAPO 소프트 게이트 (Qwen3-VL) | `--loss_type sapo` + `--tau_pos --tau_neg` | 구현 확인 (`grpo_trainer.py:1205`) |
| IcePop 계열 학습–추론 불일치 마스킹 | `--rollout_importance_sampling_mode token_mask` + `--rollout_importance_sampling_threshold` | 구현 확인 (`:2410`) |
| **불일치 계측만** | `--log_rollout_offpolicy_metrics true` | 구현 확인 (`:2502`) — KL·k3_kl·ppl_ratio·chi2_token·chi2_seq |
| GSPO 시퀀스 단위 IS | `--importance_sampling_level sequence` | 이미 사용 중(recipe gspo) |
| std 정규화 생략 (EXAONE 4.5) | `--scale_rewards none` | 이미 사용 중(recipe dr_grpo) |
| zero-variance 그룹 처리 | `--dynamic_sample true` | **이미 켜져 있음** — 폐기가 아니라 **재샘플**(DAPO 방식) |
| 엔트로피 상위 토큰만 학습 | `--top_entropy_quantile` | 미사용 |
| off-policy 시퀀스 마스킹 | `--off_policy_sequence_mask_delta` | 미사용 |

> ⚠️ EXAONE 4.5 는 zero-variance 그룹을 **버린다**. 우리는 `dynamic_sample` 로 **재샘플**한다.
> 우리 쪽이 표본을 더 쓰는 대신 롤아웃 비용이 든다. `frac_reward_zero_std` 평균 0.0106 이라
> 어느 쪽이든 영향이 미미하다. **바꿀 이유 없음.**

---

## 3. 변경 항목 상세

### A. 정확도 보상을 형식에서 분리 ✅ 완료 (`2a0458a`)

**근거** — 서베이 ①패턴: 조사한 모델 중 **정확도 보상을 형식 태그에 물려둔 사례가 없다.**
[HyperCLOVA X 32B Think](https://arxiv.org/html/2601.03286v1) 는 `<think>` 를 채팅 템플릿에 두고
format 을 언어일관성·반복억제와 같은 급의 보조항으로 취급한다.
[GLM-4.1V](https://arxiv.org/pdf/2507.01006) 는 `r_format ∈ {0, 0.5}` 대 `r_accuracy ∈ {0, 1}` 로 **독립 항**이다.

**우리 문제** — `_strip_answer` 가 태그 없을 때 추론 전문을 답안으로 넘기고,
`_LETTER_PICK` 이 `^` 앵커라 letter 경로가 **구조적으로 항상 0** 이었다.
의료(letter) −47.5pp 대 수학(본문 숫자 추출 가능) −6.5pp 로 갈린 원인.

**실측** (73924 롤아웃 9,504건, `scripts/probe_answer_fallback.py`)

| | 값 |
|---|---|
| 태그 인위 제거 후 복원 일치 | **94.6%** (거짓양성 1.13%) |
| 순열검정 | z = 3.6 / 4.9 / 6.6 — 우연 아님 |
| 회귀: 태그 있는 롤아웃 6,633건 | **전부 무변경** (완전한 no-op) |

---

### B. 형식 보상을 이진 → 단계형 🔴 신규

**근거** — [GLM-4.1V](https://arxiv.org/pdf/2507.01006) 의 `r_format ∈ {0, 0.5}` 가 이미 이진이 아니다.
다만 **직접적 근거는 우리 실측**이다.

**우리 관찰** — 붕괴 개시 시점 출력의 실제 모양:

```
"...This corresponds to option B.\n</think><|im_end|>"
```

**`</think>` 는 냈다. `<answer>` 만 없다.** 추론을 끝내고 답 태그를 안 쓰고 멈춘다.

| 구간 | n | fmt=1.0 | **`</think>`만** | 둘 다 없음 |
|---|---:|---:|---:|---:|
| 붕괴전 751-898 | 4,736 | 93.1% | 0.3% | 6.7% |
| **개시 899-910** | 384 | 10.7% | **58.9%** | 29.7% |
| 확산 911-950 | 1,280 | 22.0% | 32.1% | 37.1% |

**개시 구간의 58.9% 가 `</think>` 를 냈는데 형식 점수 0.0 을 받는다.** 절벽의 상당 부분이 여기다.

**변경안** — `configs/accuracy.py: FormatThink`

```
1.0  <think>실질추론</think><answer>…</answer>   (현행 유지)
0.5  </think> 는 냈으나 <answer> 없음            ← 신규
0.0  구조 자체가 없음                            (현행 유지)
```

**실측 효과** (총보상 = acc×1.0 + fmt×0.2 + soft_overlong×0.2)

| 구간 | 현행 | +A | **+A+B** | 낙폭 완화 |
|---|---:|---:|---:|---:|
| 붕괴전 751-898 | 0.6180 | 0.6223 | **0.6225** | (기준선, +0.7%) |
| **개시 899-910** | −0.3240 | −0.2537 | **−0.1948** | **39.9%** |
| 확산 911-950 | −0.3733 | −0.3069 | −0.2748 | 26.4% |
| 퇴화 951-1047 | −0.2323 | −0.2033 | −0.1713 | 26.3% |

> ✅ **정상 구간이 +0.0045 밖에 안 움직인다.** 붕괴 영역에만 듣는다 — 이게 안전 속성이다.
>
> ⚠️ **보상 해킹 위험**: 부분점수를 주면 `<answer>` 를 안 쓰는 게 덜 손해다. 다만 0.5 는
> 여전히 **절반 손실**이고, A 의 폴백도 의도적으로 불완전(개시 구간 정확도 0.469→0.347)하다.
> 형식을 잃으면 **총보상의 32% 를 잃는다**(−0.195 / 0.618). 유인은 남는다.
> **검증**: 재실행 로그에서 `rewards/FormatThink/mean` 이 0.9 아래로 추세 하락하면 되돌린다.

---

### C. `overlong_filter` → false

**근거 = 자체 실측** (사후분석 §4-4). 서베이 근거 없음 — DAPO 원 논문의 의도와 다르게 우리 상황에서 해롭다.

ms-swift 는 잘린 completion 을 `completion_mask` 에서 빼는데(`grpo_trainer.py:1132`),
그 마스크가 **KL 계산에도 그대로 쓰인다**(1140행). 절단이 늘수록 손실·KL 앵커 양쪽에서 면제되는
샘플이 늘어 **되돌아올 힘이 사라진다.** 부작용으로 KL 계측 자체가 둔해진다 — step 910~949 의
KL 하락(0.081→0.068)은 개선이 아니라 **계측 실패**였다.

> ⚠️ **F(불일치 계측)와 상호작용한다.** `overlong_filter=True` 면 rollout logprob 정렬 대상에서
> 절단 샘플이 빠져 계측이 왜곡된다. **C 를 먼저 꺼야 F 가 신뢰할 수 있다.**

---

### D. `scale_rewards` gdpo → none

**근거** — [EXAONE 4.5](https://arxiv.org/abs/2604.08644) 는 GRPO 를 쓰되
"**표준편차 정규화를 생략해 학습 안정성을 보존**"한다고 명시한다. Dr.GRPO 의 주장과 같다.

**실측 효과: 없음.** 사후분석 §4-2(a) 에서 gdpo 는 **무죄로 판정**됐다 — 표본 내 재가중 ±15%,
관측된 병리적 그룹 유형은 `none` 대비 1.15배에 불과. **이건 위생 조치지 치료가 아니다.**

recipe `dr_grpo` 의 기본값이 이미 `none` 이므로 `SCALE_REWARDS` 를 넘기지 않으면 된다.

---

### E. `max_steps` 를 실제 정지 step 으로

**근거 = 설계 오류.** 서베이·가설 불필요.

`--max_steps` 는 정지 지점이자 **cosine LR 스케줄의 분모**다. 2,337 로 잡고 1,200 에서 멈추면
LR 이 끝까지 안 내려온다. step 900 기준 6.77e-6 대 1.46e-6, **4.62배** 차이.

**재실행에서 실제로 돌릴 step 수를 그대로 넣는다.**

---

### F. rollout logprob 불일치 계측 🔴 신규

**근거** — 서베이 ③패턴. 2026년 불안정의 **가장 많이 지목된 원인**이다.

| 기법 | 채택 | 핵심 |
|---|---|---|
| [IcePop](https://ant-ling.medium.com/ring-flash-2-0-4bf5b62204f5) | EXAONE 4.5 | 학습·추론 토큰 확률차가 **5% 넘으면 학습 실패**. GRPO 는 **180~200 step 에서 붕괴** |
| [SAPO](https://arxiv.org/html/2511.20347v1) | Qwen3-VL | 하드 클리핑 → 온도 제어 소프트 게이트 |
| per-token reg | [Kimi K3](https://arxiv.org/abs/2607.24653) | 극단 off-policy 흡수 |

**우리 노출** — vLLM colocate + LoRA 구성은 롤아웃(추론)과 업데이트(학습)가 서로 다른 수치 경로를
탄다. 붕괴가 step 899 로 180~200 과는 다르지만, **"불일치 누적이 임계를 넘으면 절벽"이라는 형태는
관찰과 부합한다.** 다만 **우리는 이걸 한 번도 잰 적이 없다.**

**변경안 — 1차는 계측만 켠다. 보정은 켜지 않는다.**

```bash
--log_rollout_offpolicy_metrics true
```

기록되는 지표(`grpo_trainer.py:2502`): `kl`(KL(π_rollout‖π_train)) · `k3_kl` ·
`training_ppl` · `rollout_ppl` · `log_ppl_diff` · `ppl_ratio` · `chi2_token` · `chi2_seq`

> **왜 보정을 안 켜는가**: 보정(`rollout_importance_sampling_mode`)을 같이 켜면 변수가 하나 더
> 늘어 붕괴가 안 나도 무엇 덕분인지 모른다. **먼저 재고, 5% 임계를 넘는 게 확인되면 그때 켠다.**
> 켤 때는 `token_mask`(IcePop 과 같은 기전 — 임계 초과 토큰의 업데이트를 0으로) + threshold 2.0.

**전제 조건**: `use_fast_infer`(vLLM) 필요 — 충족. 정렬 실패 시 경고 후 자동 스킵하므로 안전하다.

---

### G. 감시자 상시 가동

**근거 = 자체 검증.** `scripts/watch_format_collapse.py` — 과거 로그 재생에서 **step 901 발화,
1,047 step 전 구간 오경보 0회**, 실제 발견보다 **146 step(13.5h·109 GPU-h)** 빠름.

`WATCHDOG=1` 로 21 스크립트에 이미 배선돼 있다. 재현 실험(74583)에서 오발화 0 으로 실전 검증됐다.

**같이 정할 것 — 발화 시 정책.** 지금 정해두지 않으면 새벽에 울렸을 때 판단이 늦는다.

| 안 | 동작 | 언제 |
|---|---|---|
| 정지 | 잡을 세우고 마지막 정상 체크포인트 확정 | 기본값 (현재 구현) |
| 되감기 | 마지막 정상 ckpt 에서 LR ↓ 재개 | 붕괴가 반복될 때 |

---

### H. 길이를 소프트 감점 → 하드 예산 (2차 방어선)

**근거** — [Kimi K3](https://arxiv.org/abs/2607.24653): 콜드스타트 모델에서 문제별 초기 예산
`b₀(x)` 를 추정하고, 배수 σ 초과 시 **이진 비교에서 자동 패**. 토큰 임계 초과 시 **보상 −1 로 덮어씀**.
[HyperCLOVA X 32B Think](https://arxiv.org/html/2601.03286v1) 는 **길이제어 RLVR 을 독립 단계**로 둔다.

**우리 상태** — `soft_overlong` 가중치 0.2. 길이가 1,376 → 3,010 으로 폭주하는 동안
이 항은 −0.104 수준에 머물렀다. **하드 예산이었다면 −1 로 즉시 끊겼다.**

⚠️ **다만 우선순위는 낮다.** 사후분석에서 확정했듯 **길이는 원인이 아니라 결과**다
(형식 붕괴 899~904, 길이 폭주 905~ — 10 step 뒤). 2차 방어선으로만 의미가 있다.

**A+B+C 를 먼저 넣고, 그래도 길이가 폭주하면 그때 도입한다.**

---

### I. `loss_type=sapo` ⚠️ 보류

**근거는 강하다** — [SAPO](https://arxiv.org/html/2511.20347v1) 는 Qwen3-VL 학습에 쓰였고,
GSPO·GRPO 대비 우위가 보고됐다. 비대칭 온도 `τ_neg > τ_pos` 가 안정성의 핵심(어블레이션 확인).
ms-swift 에 구현돼 있다(`tau_pos=1.0`, `tau_neg=1.05` 기본값).

**그런데 소스를 읽어보니 공짜가 아니다.**

```python
# grpo_trainer.py:1245
if self.loss_type in ['grpo', 'sapo']:
    loss = ((per_token_loss * completion_mask).sum(-1)
            / completion_mask.sum(-1).clamp(min=1.0)).mean()      # ← 시퀀스별 길이 정규화
elif self.loss_type == 'dr_grpo':
    loss = (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)
```

- **`sapo` 를 켜면 `dr_grpo` 가 제거한 길이 정규화 편향이 되돌아온다.** 둘을 동시에 못 쓴다.
- **`sapo` 분기에는 클리핑이 없다** — `epsilon` / `epsilon_high` 가 무시된다.

**판단**: 우리 recipe 는 `dr_grpo` 를 **의도적으로** 골랐다(길이 정규화 편향 제거).
길이 폭주를 겪은 실행에서 그 편향을 되돌리는 건 **검증 없이 할 일이 아니다.**

**보류하되 A/B arm 후보로 남긴다.** 소프트 게이트만 취하고 dr_grpo 정규화를 유지하려면
`grpo_trainer.py:1245` 에 `sapo`를 빼고 별도 분기를 추가하는 **10줄 패치**가 필요하다.
업스트림 수정이므로 컨테이너 재빌드 불가 상황에서는 부담이 있다.

---

## 4. 새 recipe 정의 — `stable`

기존 recipe(`dapo`/`gspo`/`dr_grpo`)를 건드리지 않고 하나 추가한다. A/B 대조가 깨지지 않는다.

```bash
# scripts/21_rlvr_grpo_adv.slurm — RECIPE=stable
CORE_ARGS="--dynamic_sample true --max_resample_times $DS_RESAMPLE \
           --overlong_filter false"                       # ← C

RECIPE_ARGS="--loss_type dr_grpo \
  --importance_sampling_level token \
  --epsilon ${EPS:-0.2} \
  --scale_rewards ${SCALE_REWARDS:-none} \
  --log_rollout_offpolicy_metrics true"                   # ← D, F
```

기존 대비 **순 변경 3줄**이다. 나머지(A·B)는 `configs/accuracy.py`, E·G 는 실행 인자.

**변경 요약 — 현행 대 `stable`**

| 인자 | 현행(73924) | `stable` | 항목 |
|---|---|---|---|
| `reward_funcs` / `weights` | 동일 | 동일 (1.0 / 0.2 / 0.2) | — |
| `AccuracyMix` letter 폴백 | 없음 | **있음** | A |
| `FormatThink` | 이진 | **단계형 0 / 0.5 / 1** | B |
| `overlong_filter` | true | **false** | C |
| `scale_rewards` | gdpo | **none** | D |
| `max_steps` | 2337(1,047에서 중단) | **실제 정지 step** | E |
| `log_rollout_offpolicy_metrics` | 없음 | **true** | F |
| `WATCHDOG` | 없음 | **1** | G |
| `loss_type` | dr_grpo | dr_grpo (유지) | I 보류 |
| `beta` | 0.04 | 0.04 (유지) | §6 |

---

## 5. 실행 계획

| 단계 | 내용 | 비용 | 판정 |
|---|---|---|---|
| **0** | `configs/accuracy.py` B 구현 + 9,504건 회귀 검증 | 0 (로컬) | 태그 정상 롤아웃 무변경 |
| **1** | 스모크 `MAX_STEPS=5` | ~1 GPU-h | 인자 파싱·`rollout_correction/*` 지표 기록 확인 |
| **2** | checkpoint-850 에서 재개, 200 step | ~90 GPU-h | 불일치 지표 5% 임계 · 형식 유지 |
| **3** | 2 의 결과로 F 보정 켤지 결정 → 본실행 | 예산에 따름 | — |

> **왜 850 인가**: 홀드아웃 51.52%, init 대비 +8.18pp(p<0.0001)로 **검증된 최고점**이다.
> 처음부터 다시 돌리는 것보다 800 step 을 아낀다.
>
> ⚠️ **큐 대기가 실질 비용이다.** 재현 실험은 제출 후 **5일 13시간** 대기했다.
> 단계 1·2 를 따로 넣으면 대기가 두 번 붙는다 — **같은 잡 안에서 스모크→본실행**을 잇는 편이 낫다.
>
> ⚠️ **속도 정정**: 벽시계는 `step_time`(163s)이 아니라 **~330 s/it** 이다. 재현 실험 12h 에 127 step.
> 일정은 반드시 330 기준으로 잡을 것.

---

## 6. 바꾸지 않는 것과 이유

| 항목 | 유지 이유 |
|---|---|
| `beta=0.04` (KL 유지) | HyperCLOVA X 는 **KL 을 제거**하지만 목적이 다르다 — 저쪽은 다양성 확보, 우리는 이탈 억제. 우리 붕괴는 다양성 부족이 아니었다 |
| `epsilon_high` 미설정(대칭 0.2) | [clip-low 는 엔트로피를 올리고 clip-high 는 내린다](https://arxiv.org/pdf/2509.26114). 상단 클립을 열면 **큰 상승 업데이트를 허용**하는 쪽 — 지금 원하는 방향의 반대 |
| `num_generations=4` | `frac_reward_zero_std` 평균 **0.0106** — 그룹은 끝까지 건강했다. 8로 올리면 롤아웃 비용 2배 |
| `freeze_vit=True` | Kimi K3 가 비전 타워 최적화 불안정(gradient norm 스파이크)을 보고했다. RL 에서 여는 건 별도 실험 |
| `dynamic_sample=true` | EXAONE 4.5 는 zero-variance 그룹을 버리지만 우리는 재샘플한다. 영향 미미(0.0106), 바꿀 이유 없음 |
| **모델 크기 9B** | A100 80GB **PCIe**(NVLink 없음, P2P 차단) + vLLM colocate 가중치 2벌 → 실효 상한 ~12B. 14B 는 예산 47% 소모. 그리고 **붕괴는 용량 문제가 아니었다** |

---

## 7. 이 문서가 다루지 않는 것

서베이에서 나온 **구조적 제안 2건**은 별도 판단이 필요해 여기 넣지 않았다.

**① 텍스트 전용 콜드스타트** — [ReVisual-R1](https://github.com/CSfufu/Revisual-R1)(ICLR 2026)은
텍스트 전용 콜드스타트 → 멀티모달 RL → 텍스트 RL 3단이다. 관련 연구는
**멀티모달 콜드스타트가 시각 주의를 못 올리고 텍스트 전용은 올린다**고 보고한다
([Lazy Attention Localization](https://arxiv.org/html/2603.03825v1)).
우리 Stage-1 은 멀티모달 혼합이다. **의료 여섯 지점 미검출과 정합적인 가설**이지만
Stage-1 재실행이 필요하다.

**② Stage-2/3 를 순차 → 병렬 전문가 + 증류** — [DeepSeek-V4](https://arxiv.org/html/2606.19348v1) 와
[Kimi K3](https://arxiv.org/abs/2607.24653) 가 독립적으로 도달한 구조다.
우리 데이터가 이미 순차 구조의 대가를 보여준다 — Stage-2 는 일반 +8.18pp, **의료 +0.75pp(p>0.5)**.
다만 PCIe 하드웨어에서 전어휘 KL 증류는 통신 비용이 크다.
**작게 시작하는 길** = 전문가 2개(일반·의료) + 증류 대신 가중평균 병합부터.

둘 다 **Stage-3 일정과 예산에 직접 영향**을 주므로 별도 결정 문서가 필요하다.

---

## 8. 레퍼런스

**테크니컬 리포트**
- [EXAONE 4.5 Technical Report (arXiv 2604.08644)](https://arxiv.org/abs/2604.08644) — GRPO+IcePop, zero-variance 필터, **std 정규화 생략**
- [K-EXAONE 2.0 (arXiv 2608.04505)](https://arxiv.org/abs/2608.04505) — format-invariant tool-use, Clamped SwiGLU
- [HyperCLOVA X 32B Think (arXiv 2601.03286)](https://arxiv.org/html/2601.03286v1) — `<think>` 를 템플릿에, format 은 보조 보상, 길이제어 RLVR 독립 단계
- [Kimi K3 (arXiv 2607.24653)](https://arxiv.org/abs/2607.24653) — 길이 하드 예산, per-token 정규화, 비전 타워 불안정
- [DeepSeek-V4 (arXiv 2606.19348)](https://arxiv.org/html/2606.19348v1) — 도메인 전문가 + 전어휘 KL 증류
- [GLM-4.5V / GLM-4.1V-Thinking (arXiv 2507.01006)](https://arxiv.org/pdf/2507.01006) — `r_format ∈ {0,0.5}` 대 `r_accuracy ∈ {0,1}`

**알고리즘**
- [Soft Adaptive Policy Optimization — SAPO (arXiv 2511.20347)](https://arxiv.org/html/2511.20347v1)
- [IcePop / Ring-flash-2.0](https://ant-ling.medium.com/ring-flash-2-0-4bf5b62204f5) · [Every Step Evolves (arXiv 2510.18855)](https://arxiv.org/pdf/2510.18855)
- [Clip-Low Increases Entropy and Clip-High Decreases Entropy (arXiv 2509.26114)](https://arxiv.org/pdf/2509.26114)
- [DAPO (arXiv 2503.14476)](https://arxiv.org/abs/2503.14476) · [Dr. GRPO (arXiv 2503.20783)](https://arxiv.org/abs/2503.20783) · [GSPO (arXiv 2507.18071)](https://arxiv.org/abs/2507.18071)

**자체 문서**
- [2026년 모델 서베이](rlvr_survey_2026.md) — 이 설계의 외부 근거
- [붕괴 사후분석 rev.2](stage2_run73924_postmortem.md) — C·D 항목의 실측 근거
- `scripts/probe_answer_fallback.py` — A·B 항목의 실측 도구
