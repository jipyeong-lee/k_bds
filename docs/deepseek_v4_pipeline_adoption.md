# DeepSeek-V4 파이프라인 도입안 — 도메인별 전문가 + 통합

> 작성 2026-08-13 · **rev.2** — 도메인 분리 학습으로 전면 개정
> 근거 = [DeepSeek-V4 (arXiv 2606.19348)](https://arxiv.org/html/2606.19348v1) ·
> [Kimi K3 (arXiv 2607.24653)](https://arxiv.org/abs/2607.24653) ·
> [2026년 모델 서베이](rlvr_survey_2026.md) · ms-swift 4.13 설치본 소스 직접 확인
>
> 🚨 **rev.1 정정**: 초판은 "전문가 A = 기존 혼합 실행(ck-850)"으로 잡았다. **그건 DeepSeek-V4
> 구조가 아니다.** 혼합 학습된 모델은 도메인 전문가가 아니라 그냥 혼합 모델이다.
> 이 개정판은 **Stage-2 자체를 도메인별로 쪼갠다.** 대가도 함께 적는다 — ck-850 을 공짜
> 전문가로 쓰던 이점이 사라진다.

---

## 0. 리포트에서 확인된 것과 안 된 것

**DeepSeek-V4 리포트는 후처리 부분이 얇다.** 시작 전에 알고 있어야 한다.

| 항목 | 리포트 |
|---|---|
| 2단 구조 | ✅ "독립적 도메인 전문가 육성 → 온폴리시 증류로 통합" |
| 도메인 | ✅ 수학·코딩·에이전트·지시따르기 |
| 전문가 학습법 | ✅ 도메인별 SFT → **GRPO** + 도메인별 보상 |
| 통합 손실 | ⚠️ "reverse KL"·"Full-Vocabulary OPD" 표현만. **수식 없음** |
| 교사 가중·하이퍼파라미터 | ❌ 없음 |
| **전문가 대 통합모델 비교 수치** | ❌ **없음** |

> 세부는 같은 구조(MOPD)를 **수식까지 공개한** [Kimi K3](https://arxiv.org/abs/2607.24653) 를 참조한다.

---

## 1. 도메인 분리가 왜 옳은가 — 실측

현재 Stage-2 는 세 소스를 **한 데 섞어** 학습한다. 노출량을 계산했다
(`work/data/stage2_expanded_train.jsonl`, 프롬프트/step = 8):

| 소스 | 건수 | 비중 | 1 epoch | **혼합 850 step 시 노출** |
|---|---:|---:|---:|---:|
| deepvision (일반) | 40,000 | 53.5% | 5,000 step | 3,636건 = **0.09 epoch** |
| pmcvqa (의료) | 19,583 | 26.2% | 2,448 step | 1,780건 = **0.09 epoch** |
| mmk12 (수학) | 15,204 | 20.3% | 1,900 step | 1,382건 = **0.09 epoch** |

**같은 step 수를 도메인 전용으로 쓰면 노출이 이만큼 늘어난다:**

| 소스 | 전용 600 step | 전용 850 step |
|---|---:|---:|
| deepvision | 0.12 epoch (**1.3배**) | 0.17 epoch (**1.9배**) |
| pmcvqa | 0.245 epoch (**2.7배**) | 0.35 epoch (**3.8배**) |
| mmk12 | 0.316 epoch (**3.5배**) | 0.45 epoch (**4.9배**) |

**의료는 같은 계산자원으로 3.8배, 수학은 4.9배 더 본다.** 소수 도메인일수록 이득이 크다 —
혼합 학습에서 그래디언트가 다수 도메인(deepvision 53.5%)에 지배되던 것이 사라진다.

### ⚠️ 다만 이 논거만으로 의료 실패를 설명할 수는 없다

정직하게 적는다. 홀드아웃 init→850 결과:

| | deepvision | mmk12 | **pmcvqa** |
|---|---:|---:|---:|
| init | 35.60% | 48.25% | **57.25%** |
| init→850 | **+10.80** ✅ | **+9.25** ✅ | **+0.75** ✗ (p>0.5) |
| 혼합 노출 | 0.09 epoch | 0.09 epoch | 0.09 epoch |

**수학은 의료와 완전히 같은 0.09 epoch 노출로 +9.25pp 를 얻었다.**
따라서 "노출 부족"은 의료 실패의 **충분한 설명이 아니다.** 경쟁 가설:

- **여유 폭 차이** — 의료는 init 57.25% 로 셋 중 가장 높다. 남은 오답이 더 어렵다.
- **지식 결손 대 추론 결손** — RLVR 은 **있는 능력을 끌어내지 없는 지식을 넣지 못한다.**
  의료 오답이 지식 부족이면 의료 RL 을 아무리 해도 안 오른다.
  ([Stage-3 문서 §base 프로파일](stage3_and_eval.md): 맥락인지 0.10·완결성 0.24 — 이게 RaR 이 겨냥하는 결함)

> **결론: 도메인 분리는 옳다. 다만 "이걸 하면 의료가 오른다"고 약속할 수 없다.**
> 3.8배 노출은 **가설을 제대로 검정할 기회**를 주는 것이지 결과를 보장하지 않는다.
> 지금은 노출이 너무 적어 **의료 RL 이 안 듣는 건지 덜 한 건지 구분조차 안 된다.**

### Kimi K3 의 경고 — 너무 잘게 쪼개지 말 것

> "Rather than training specialized RL models for individual tasks, we scale RL across
> **three broad domains**." — [Kimi K3](https://arxiv.org/abs/2607.24653)

Kimi 는 **일부러 넓은 도메인 3개**로 묶었다. 우리 소스 3개는 그 정도 입도에 맞는다.
**더 쪼개지 않는다.**

---

## 2. 새 구조

### 발견: 의료에 데이터셋이 둘이다

`scripts/30_medical_rl.slurm` 을 확인한 결과 Stage-3 는 **pmcvqa 가 아니다.**

| | Stage-2 의료 | Stage-3 의료 |
|---|---|---|
| 데이터 | `stage2_expanded_train.jsonl` 중 pmcvqa 19,583 | `medix_rl_train.jsonl` (단답 VQA, 정답 중앙값 46자) |
| 보상 | `accuracy_mix`(letter) + format + soft_overlong | `clinical_judge`(RaR 루브릭, 1.0) + `format_think`(0.2) |
| 채점 | 규칙 기반 검증 | 외부 멀티모달 judge (Qwen3.6-27B-FP8) |

**둘은 다른 능력을 겨냥한다** — pmcvqa 는 객관식 정답률, medix 는 개방형 임상 서술.
따라서 하나의 "의료 전문가"로 합칠 수도, 둘로 나눌 수도 있다.

### 권고 구성 — 전문가 3 + 의료 2단

```
sft_mixed_merged ──┬─→ [E1 일반]  deepvision 40,000 · accuracy_mix + format
   (동결 base,     │
    LoRA r16)      ├─→ [E2 수학]  mmk12 15,204 · math_verify 경로 · 길이 예산 ↑
                   │
                   └─→ [E3 의료]  pmcvqa 19,583 (RLVR·letter)
                            └─→ medix (RaR·clinical_judge)   ← Stage-3 가 여기 흡수된다
                                                                
  통합: E1 + E2 + E3 → 단일 학생 (계단 0 어댑터 산술 → 계단 1 GKD)
```

> 🔴 **Stage-3 가 독립 단계에서 E3 의 2차 학습으로 바뀐다.** 현재 Stage-3 계획은
> "Stage-2 체크포인트 병합 → init 교체"이고 문서에 **"망각 방지용 DeepVision 혼합은 옵션"**
> 이라 적혀 있다. 그 망각 문제가 **구조적으로 사라진다** — E3 는 일반 능력을 지킬 의무가 없고,
> 일반 능력은 E1 이 들고 있다가 통합 때 합쳐진다.

### 도메인 분리로 새로 생기는 손잡이

혼합 학습에서는 불가능했던 것들이다. **이게 노출량 다음으로 큰 이득이다.**

| 손잡이 | E1 일반 | E2 수학 | E3 의료 |
|---|---|---|---|
| 길이 예산(`soft_max_length`) | 중간 | **크게** — 수학은 긴 CoT 가 필요 | **작게** — 객관식 |
| `num_generations` | 4 | 4 | 필요시 ↑ (여유 폭이 좁아 변별 필요) |
| 보상 가중 | 기본 | 기본 | 형식 비중 조정 여지 |
| 조기 종료 | 독립 판단 | 독립 판단 | 독립 판단 |

---

## 3. 실현 가능성 — ms-swift 4.13

설치본 `swift/rlhf_trainers/gkd_trainer.py`(1,071줄) 직접 확인.

### 되는 것

| 필요 | 인자 | 확인 |
|---|---|---|
| 온폴리시 증류 | `--rlhf_type gkd` | `rlhf_args.py:229` |
| 학생 자기 롤아웃 | `--lmbda 0.5` | `gkd_trainer.py:491` |
| **전어휘** KL/JSD | `generalized_jsd_loss`, top-k 미지정 시 full-vocab | `:774`, `:231` |
| forward/reverse KL 보간 | `--beta` (0=forward, 1=reverse) | `:838` |
| **멀티모달** | `is_multimodal` 분기 + vLLM 생성 | `:257`, `:629` |
| LoRA 자기증류 최적화 | `disable_adapter()` — 교사 사본 없음 | `:88`, `:406` |
| 원격 교사 | `--teacher_model_server` (멀티모달 지원) | `:615` |

### 갭 두 개 ⚠️

**① 다중 교사 미지원** — `rlhf_type=gkd` 는 교사를 하나만 받는다.

**② `teacher_adapters` 미배선** — 인자는 있으나(`rlhf_args.py:68`) 교사 구성 지점이 다른 걸 읽는다:

```python
# pipelines/train/rlhf.py:94
adapters = args.adapters if key == 'ref' else args.reward_adapters
model = prepare_adapter(args, model, adapters)      # key=='teacher' 여도 reward_adapters
```

> 전문가 LoRA 를 교사로 올릴 때 **`--teacher_adapters` 가 아니라 `--reward_adapters`** 를 쓴다.
> 스모크로 반드시 확인할 것.

---

## 4. 전이되지 않을 수 있는 것 — 정직하게

- ⚠️ **규모 차이.** DeepSeek-V4 는 1.6T MoE(49B 활성). 우리 학습 파라미터는 **43M(9B의 0.48%)**.
  전문가 분리는 **용량이 충분할 때** 간섭을 푸는 방법이다. 같은 이득 크기를 기대할 근거는 없다.
- ⚠️ **도메인 이질성이 훨씬 작다.** DeepSeek 의 수학/코딩/에이전트/지시는 행동 자체가 다르다.
  우리 셋은 전부 "이미지 보고 추론해 객관식/단답". **간섭이 애초에 작았을 수 있고**(분리 이득 ↓),
  대신 **병합이 쉬울 수 있다**(계단 0 이득 ↑).
- 🚨 **통합 용량.** 어댑터 3개의 행동을 같은 rank 16 공간에 넣어야 한다.
  → **학생 rank 를 32~48로 올린다.** 메모리 비용 무시 가능. 우리 상황에만 있는 손잡이다.
- ⚠️ **원저자도 전문가 대 통합모델 비교를 공개하지 않았다.** 통합의 손실률은 **우리가 직접 잰다.**

---

## 5. 이 개정의 대가 — ck-850 을 잃는다

rev.1 에서 강조했던 "전문가 A 가 공짜로 있다"가 **사라진다.**
`checkpoint-850` 은 혼합 학습된 모델이지 도메인 전문가가 아니다.

| | rev.1 (혼합 A + 의료 B) | **rev.2 (도메인 3분할)** |
|---|---|---|
| 전문가 A 비용 | **0** (ck-850 재사용) | **623 GPU-h** (E1 신규 학습) |
| 도메인 노출 | 0.09 epoch | **1.3~4.9배** |
| 도메인별 하이퍼파라미터 | 불가 | **가능** |
| 구조 충실도 | 절반 | 완전 |

**ck-850 을 완전히 버릴 필요는 없다.** 두 가지 쓸모가 남는다:

1. **기준선** — 통합 모델이 ck-850(홀드아웃 51.52%)을 넘는지가 이 구조 변경의 판정 기준이다.
2. **4번째 교사 후보** — 혼합 모델이라 "일반화된 행동"을 들고 있다. 계단 1 에서 섞어볼 수 있다.

---

## 6. 비용 — 전문가별 1 epoch 은 가능한가

기준: 벽시계 **330 s/it**(재현 실험 74583 실측) · 1 벽시계시 = **8 GPU-h**

**실측 예산 잔량** (`sacct -u k252a02`, 8gpu 파티션 누적 109.3 벽시계h):

```
사용 874 GPU-h / 5,000  →  잔여 4,126 GPU-h (82.5%)
```

**파티션 제약** (`scontrol show partition 8gpu`): `MaxTime=5-00:00:00` · `TotalNodes=3`
→ 잡 하나가 최대 120 벽시계h = **1,309 step**. 그 이상은 잡을 연쇄해야 한다.

### 1 epoch × 3 = 예산의 166%. 불가능하다

| 전문가 | 건수 | 1 epoch | 벽시계h | GPU-h | 잔여 대비 | 필요 잡 수 |
|---|---:|---:|---:|---:|---:|---:|
| E1 deepvision | 40,000 | 5,000 step | 458 | 3,667 | 88.9% | **4 연쇄** |
| E2 mmk12 | 15,204 | 1,900 step | 174 | 1,394 | 33.8% | 2 연쇄 |
| E3 pmcvqa | 19,583 | 2,448 step | 224 | 1,795 | 43.5% | 2 연쇄 |
| **합계** | 74,787 | **9,348 step** | 857 | **6,856** | **166%** 🚨 | **8 연쇄** |

두 가지가 동시에 막는다.

- **예산**: 6,856 > 4,126. Stage-3(medix RaR)·통합·평가에 쓸 것이 하나도 안 남는다.
- **큐**: 8개 잡 연쇄. 관측된 대기가 **5일 13시간**이라 E1 체인만 대기 22일이다.

### 그래서 — 어느 도메인에 1 epoch 을 줄 것인가

전부는 안 된다. **하나만 고른다면 의료다.**

- 의료가 **계획서의 산출물**이고
- **유일하게 실패한 도메인**이며(+0.75pp, p>0.5)
- 지금 노출(0.09 epoch)로는 **안 듣는 건지 덜 한 건지 구분조차 안 된다**

일반·수학은 이미 각각 +10.80 / +9.25pp 로 **듣는다는 게 입증됐다.** 더 태울 이유가 약하다.

### E1 에 관한 실측 판단 — ck-850 이 285 step 어치 앞서 있다

ck-850 이 본 deepvision = 850 × 8 × 0.535 = **3,638건**.
전용 E1 이 이를 따라잡으려면 **455 step**(3,640건)이 필요하다.

> **즉 E1 을 455 step 미만으로 돌리면 ck-850 보다 deepvision 을 덜 본 모델이 나온다.**
> 그 구간에서는 **ck-850 을 일반 교사로 쓰는 편이 낫다.** 다만 ck-850 은 혼합 학습본이라
> 순수 도메인 전문가가 아니다 — 통합 시 그만큼 도메인 경계가 흐려진다.

### 권고 배분

| | 전문가 | step | epoch | GPU-h | 잡 |
|---|---|---:|---:|---:|---:|
| **A안 (권장)** | E3 pmcvqa | **2,448** | **1.00** | 1,795 | 2 |
| | E2 mmk12 | 950 | 0.50 | 697 | 1 |
| | E1 = **ck-850 재사용** | — | — | 0 | 0 |
| | **소계** | | | **2,492** | **3** |
| | 잔여(Stage-3 medix·통합·평가·예비) | | | **1,634** | |
| **B안** | E3 pmcvqa | 2,448 | 1.00 | 1,795 | 2 |
| | E2 mmk12 | 1,900 | 1.00 | 1,394 | 2 |
| | E1 deepvision | 600 | 0.12 | 440 | 1 |
| | **소계** | | | **3,629** | **5** |
| | 잔여 | | | 497 🚨 | |

> **A안 권장.** 의료에 1 epoch 을 온전히 주고, 나머지 1,634 GPU-h 로 Stage-3 medix RaR ·
> 통합 · 평가를 감당한다. B안은 잔여 497 GPU-h 로 Stage-3 를 못 돌린다.
>
> E2 를 0.5 epoch 으로 잡은 근거: 수학은 혼합 0.09 epoch 에서 이미 +9.25pp 를 얻었다.
> **0.5 epoch 은 그 5.5배**다. 한계효용이 남아 있는지 보기엔 충분하고, 부족하면 이어학습한다.

### 병렬 실행

노드가 정확히 3개다. **A안은 잡 3개**라 한 번에 띄우면 대기가 한 번이다.
다만 파티션 전체 점유이므로 **다른 사용자와의 충돌 확인 필요.** 2개씩 나누면 대기가 두 번.

---

## 7. LoRA 설정 점검

현재 (`checkpoint-850/adapter_config.json` + 실행 인자):

```
tuner_type      lora           lora_dropout    0.05      ← 🔴 문제
lora_rank       16             use_rslora      False
lora_alpha      32 (= 2r)      lora_dtype      None (bf16)
target_modules  all-linear → 실제로는 model.language_model.* 만
                (freeze_vit=True · freeze_aligner=True 로 ViT·aligner 제외)
어댑터           173 MB ≈ 43M 파라미터 = 9B 의 0.48%
```

### 🔴 `lora_dropout` 0.05 → 0 으로. RL 에서는 꺼야 한다

**근거는 소스다.**

```python
# grpo_trainer.py:229 — 생성 직후 모델을 명시적으로 train() 으로 되돌린다
if mode == 'train':
    self.model.train()

# :941 — logprob 계산. no_grad 지만 dropout 은 grad 모드가 아니라 module.training 을 본다
with torch.no_grad(), disable_gradient_checkpointing(...):
    batch_encoded_inputs['old_per_token_logps'] = ...
```

**vLLM 롤아웃에는 dropout 이 없다.** 따라서 `π_rollout ≠ π_train` 이 **구조적으로** 발생한다.
서베이 ③이 지목한 학습–추론 불일치와 같은 계열인데, **이건 우리가 스스로 만든 것**이다.

그리고 [재설계 문서](stage2_redesign_2026.md) F 항목(불일치 계측)을 오염시킨다 —
dropout 잡음과 실제 vLLM/HF 수치 격차를 **구분할 수 없게 된다.**

> ⚠️ **정확히 하자면 중요도 비율(ratio)은 영향이 없다.** `old_policy()`(`:1927`)가
> `num_iterations=1` 이고 `grad_accum(4) % steps_per_generation(4) == 0` 이라 **False** 를
> 반환해, `old_per_token_logps` 를 따로 구하지 않고 `per_token_logps.detach()` 를 쓴다(`:1152`).
> 같은 forward 라 비율은 정확히 1이다.
> **문제는 비율이 아니라 ① 최적화되는 정책과 샘플링한 정책의 괴리 ② KL 항 ③ 그래디언트 분산**이다.
>
> **비용 0. 도메인 전문가 3개 모두 `--lora_dropout 0` 으로 간다.**

### rank — 전문가는 16 유지, 통합 학생만 올린다

| | rank | 이유 |
|---|---|---|
| 전문가 E1·E2·E3 | **16 유지** | 🔴 **어댑터 산술 병합(계단 0)은 rank 가 같아야 한다.** 하나라도 다르면 §8 계단 0 이 불가능해진다 |
| 통합 학생 (계단 1 GKD) | **32~48** | 전문가 3개의 행동을 한 공간에 넣어야 한다. 메모리 비용 무시 가능 |

> rank 를 올릴 때는 **`--use_rslora true`** 를 같이 켠다 — alpha 스케일이 `1/r` 대신 `1/√r` 이라
> 고랭크에서 학습률이 과도해지지 않는다. 현재 `use_rslora=False` 는 r=16 에서는 문제없다.

### 그대로 두는 것

| 항목 | 판단 |
|---|---|
| `lora_alpha=32` (= 2r) | 표준. 유지 |
| `target_modules=all-linear` | 실제 해석이 `model.language_model.*` 로 정확히 의도대로다. 유지 |
| `freeze_vit=True` | [Kimi K3](https://arxiv.org/abs/2607.24653) 가 비전 타워 최적화 불안정(gradient norm 스파이크)을 보고했다. RL 에서 여는 건 별도 실험 |
| `lora_dtype=None` | bf16 을 따라간다. 유지 |

---

## 8. 계단식 통합 — 공짜부터

전문가 셋이 **같은 base·rank·target_modules** 의 LoRA 이므로 **어댑터 산술이 가능하다.**
(`checkpoint-850/adapter_config.json` 확인: r=16, alpha=32, base=`sft_mixed_merged`,
target=`model.language_model.*`, 173MB ≈ 43M 파라미터)

**계단 0 — 어댑터 산술 병합 (~30 GPU-h, 학습 0)**

```
ΔW = w₁·ΔW_일반 + w₂·ΔW_수학 + w₃·ΔW_의료
```

단순 평균 → 가중 스윕 → 필요시 TIES/DARE. 홀드아웃(n=1,772)은 **소스별 층이 이미 나뉘어 있어**
(deepvision 972 / mmk12 400 / pmcvqa 400) 도메인별 보존율을 직접 잰다.

- **판정**: 각 도메인이 해당 전문가의 **90% 이상** 보존 → 여기서 끝
- **건너뛰지 말 것** — 되면 증류 비용 전체를 아끼고, 안 되어도 증류의 기준선이 된다

**계단 1 — 순차 GKD** (재고 ms-swift). 위험 = 뒤 증류가 앞을 잊음. 완화 = 데이터 혼합 + λ 하향.

**계단 2 — 도메인 라우팅 다중 교사** (패치 필요). 샘플의 이미지 경로로 소스를 알 수 있으므로
(`scripts/train_source_trend.py:74` `source_of()`) 교사 스왑 로직은 수십 줄이다.
apptainer 파손 상태라 site-packages 직접 수정 가능 여부 확인 필요.

---

## 9. 실행 순서와 결정 사항

```
[1] 전문가 학습 설정 확정 — 도메인별 데이터 분할 + stable recipe 적용
      · stage2_expanded_train.jsonl 을 소스별 3개로 분할 (이미지 경로 기준, 스크립트 필요)
      · docs/stage2_redesign_2026.md 의 A~G 를 그대로 적용
[2] 스모크 1잡 (MAX_STEPS=5 ×3 arm) — 인자·분할·지표 확인
[3] E1·E2·E3-a 동시 제출 (가능하면 3노드 병렬)
[4] E3-b: medix RaR 을 E3-a 위에 (judge 서버 필요)
[5] 계단 0 병합 스윕 → 판정 → 필요시 계단 1
```

**지금 정해야 할 것 세 가지**

| | 선택지 | 권고 |
|---|---|---|
| **①** 도메인 3분할인가 2분할(일반+수학 / 의료)인가 | Kimi K3 는 "넓은 도메인 3개" | **3분할** — 소스가 이미 3개고 더 쪼개지 않는다 |
| **②** 전문가당 step 수 | 600 / 850 | **600 으로 시작** — 셋 합쳐 1,320 GPU-h(26%). 부족하면 이어학습 |
| **③** 파티션 3노드 동시 점유 허용 여부 | 가능 / 2개씩 | **확인 필요** — 대기 5.5일 × 횟수가 실질 비용 |

---

## 10. 레퍼런스

- [DeepSeek-V4 (arXiv 2606.19348)](https://arxiv.org/html/2606.19348v1) — 2단 구조, 전어휘 OPD. 후처리 서술은 얇음
- [Kimi K3 (arXiv 2607.24653)](https://arxiv.org/abs/2607.24653) — MOPD 수식 공개, **"넓은 도메인 3개"** 원칙
- [DeepSeek V4: ten teachers, one student](https://maximelabonne.substack.com/p/deepseek-v4-ten-teachers-one-student) — 2차 정리 (C등급)
- [2026년 모델 서베이](rlvr_survey_2026.md) §2-⑤ — 이 구조가 관행이 된 경위
- [Stage-2 재실행 실험 설계](stage2_redesign_2026.md) — 전문가 학습에 적용할 설정 A~G
- [Stage-3 · 의료 RL](stage3_and_eval.md) — E3-b 의 루브릭·judge 설계
- [붕괴 사후분석 rev.2](stage2_run73924_postmortem.md) — ck-850 기준선의 근거
- ms-swift 설치본: `work/images/ms-swift-413-sandbox/.../swift/rlhf_trainers/gkd_trainer.py`
