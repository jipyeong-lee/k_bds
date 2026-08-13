# 2026년 중국·한국 모델 학습 단계 조사 (2026-05 ~ 08 중심)

> 조사일 2026-08-13 · 목적: **Stage-2 재설계**를 위한 외부 근거 확보
> 질문 두 개 — ① 최근 모델들은 학습 단계를 어떻게 쪼개는가 ② `<think></think><answer></answer>`
> 형식을 버리면 Stage-1 콜드스타트 SFT 도 필요 없어지는가

## 0. 신뢰도 표기

이 문서의 근거는 세 등급이다. 섞어 읽으면 안 된다.

| 등급 | 뜻 | 이 문서에서 |
|---|---|---|
| **A** | arXiv 테크리포트 본문에서 직접 확인 | K-EXAONE 2.0 · EXAONE 4.5 · Kimi K3 · HyperCLOVA X 32B Think · SAPO · IcePop |
| **B** | 테크리포트 초록 + 2차 정리 글 교차 | DeepSeek-V4 · Qwen3.5-Omni · Qwen3-VL |
| **C** | 2차 출처만 (벤더 블로그·비교 사이트) | GLM-5.2 · MiniMax M3 · Kimi K3 일부 수치 |

C 등급은 방향성 참고용이다. 설계 근거로 쓰지 말 것.

---

## 1. 한눈에 보는 학습 단계

| 모델 | 공개 | 규모 | 사전학습 이후 단계 |
|---|---|---|---|
| **K-EXAONE 2.0** (LG) | 26-08-05 | 750B MoE / 37B 활성 | upcycling → healing → CPT 8T → mid-train 64K(400B) → mid-train 256K(400B) → **SFT 350B** → 선호학습① 검증가능보상 → 선호학습② 안전 |
| **EXAONE 4.5** (LG, **VLM**) | 26-04-09 | — | 비전인코더 1.2B 스크래치 → 멀티모달 사전학습 S1(420B img+400B txt) → S2(225B+110B) → **SFT** → 오프라인 선호(비전 DPO·텍스트 GROUPER) → **멀티모달 RL 합동** |
| **HyperCLOVA X 32B Think** (Naver) | 26-01 | 32B | **TSFT → MSFT** → MRLVR 4단: 도메인통합 → **길이제어** → 멀티턴 → 지시따르기 → PPO 정렬 |
| **Kimi K3** (Moonshot) | 26-07 | 2.8T MoE / 104B 활성, 네이티브 비전 | **SFT(콜드스타트)** → RL **3도메인 × 3추론강도 = 전문가 9개** → **MOPD** 다교사 온폴리시 증류 |
| **DeepSeek-V4** | 26-06 | 1.6T MoE / 49B 활성 | 도메인별 **SFT→GRPO 전문가 개별 육성** → **OPD** 전어휘 KL 증류로 단일 모델 통합 |
| **Qwen3.5-Omni** (Alibaba) | 26-04-17 | 수천억 MoE | Qwen3 계보 4단: **long-CoT 콜드스타트 → 추론 RL → thinking-mode fusion → 범용 RL**, Thinker 3단 후처리 |
| **GLM-5.2** (Zhipu) ⚠️C | 26-06 | 753B | OPD + **critic 기반 PPO 회귀** (GRPO 이탈) |
| **MiniMax M3** ⚠️C | 26-06-01 | — | 희소어텐션·네이티브 멀티모달 |

---

## 2. 횡단 패턴 — 7가지

### ① 형식 태그는 **보상에서 빠지고 채팅 템플릿으로 내려갔다** 🔴

이 프로젝트에 가장 직접적인 발견이다.

- **HyperCLOVA X 32B Think**(A): `<think>` 태그를 **채팅 템플릿에** 둔다. 비추론·추론·에이전트·멀티모달 4종 템플릿을 따로 쓴다. 보상 쪽 format 은 "**보조 항**"이며 언어일관성·반복억제와 **같은 급**으로 묶여 있다.
- **K-EXAONE 2.0**(A): 반대 방향으로 더 나갔다 — mid-training 에서 **일부러 다양한 템플릿에 노출**시켜 "**format-invariant** tool-use" 를 학습시킨다. 형식을 외우게 하는 게 아니라 형식에 **둔감**하게 만든다.
- **GLM-4.1V**(B): `r_format ∈ {0, 0.5}`, `r_accuracy ∈ {0, 1}` — 형식은 정확도의 **절반 값**이고 **독립 항**이다.

> **아무도 정확도 보상을 형식 태그에 물려두지 않는다.** 우리가 겪은 절벽 — 태그를 잃으면 가중치
> 1.0 의 정확도까지 0 — 은 2026년 설계 관행에서 벗어난 것이다. (커밋 `2a0458a` 로 이미 완화)

### ② 표준편차 정규화를 **뺀다**

- **EXAONE 4.5**(A): GRPO 를 쓰되 "**표준편차 정규화를 생략해 학습 안정성을 보존**한다". 그룹 평균만 빼고 나눗셈을 안 한다.
- 우리 실행은 `scale_rewards=gdpo`(보상함수별 z → 가중합 → 배치 화이트닝)였다. 사후분석에서 **무죄**로 판정했지만(재가중 ±15%), **필드 관행은 나눗셈을 빼는 쪽**이다.
- **EXAONE 4.5 는 zero-variance filtering 도 명시**한다 — advantage 가 전부 0 인 그룹을 버린다. 우리는 `frac_reward_zero_std` 평균 0.0106 이라 물리진 않았지만, 공짜다.

### ③ 2026년 불안정의 주범은 **학습–추론 확률 불일치**다 🔴

| 기법 | 출처 | 내용 |
|---|---|---|
| **IcePop** | Ant Ling, EXAONE 4.5 채택(A) | 학습·추론 엔진의 토큰 확률차가 **5% 넘으면 학습이 사실상 실패**. 토큰별 마스킹 + 양방향 절단으로 불일치 토큰의 업데이트를 버린다. **GRPO 는 180~200 step 에서 붕괴, IcePop 은 지속 상승** |
| **SAPO** | Qwen3-VL 채택(A, arXiv 2511.20347) | 하드 클리핑을 **온도 제어 soft gate** 로 대체. 비대칭 온도 `τ_neg > τ_pos` 가 안정성의 핵심(어블레이션 확인) |
| **per-token regularization** | Kimi K3(A) | partial rollout 이 여러 iteration 에 걸치며 생기는 극단적 off-policy 를 토큰 단위 정규화로 흡수 |

> ⚠️ **우리도 같은 계열의 위험에 노출돼 있다.** vLLM colocate + LoRA 구성은 롤아웃(추론)과
> 업데이트(학습)가 서로 다른 수치 경로를 탄다. 우리 붕괴가 **step 899, 즉 180~200 이 아니라 900 대**
> 였다는 점은 다르지만, "확률차 누적이 임계를 넘으면 절벽"이라는 형태는 관찰과 부합한다.
> **다음 실행에서 반드시 계측할 것** — 롤아웃 logprob 대 학습 logprob 의 차이 분포.

### ④ 길이는 **소프트 패널티가 아니라 하드 예산**으로 다룬다

- **Kimi K3**(A): 콜드스타트 모델에서 문제별 초기 예산 `b₀(x)` 를 추정 → 배수 σ 초과 시 **이진 비교에서 자동 패**. 토큰 임계 초과 시 **보상을 −1 로 덮어씀**.
- **HyperCLOVA X 32B Think**(A): **길이제어 RLVR 을 독립 단계**로 둔다.
- **ReVisual-R1**(ICLR 2026): Efficient-Length Reward.

> 우리 `soft_overlong` 은 가중치 0.2 의 부드러운 감점이다. 붕괴 후 길이가 1,376 → 3,010 으로
> 폭주하는 동안 이 항은 −0.104 수준에 머물렀다. **하드 예산이었다면 −1 로 즉시 끊겼다.**

### ⑤ 도메인 전문가를 **따로 키우고 증류로 합친다** 🔴

2026년의 가장 큰 구조적 변화다. **DeepSeek-V4 와 Kimi K3 가 독립적으로 같은 결론**에 도달했다.

```
        [단일 모델에 순차 RL]              [2026년 관행]
        base → SFT → RL(일반)          base → SFT ─┬→ RL(일반)   ┐
                   → RL(의료)                      ├→ RL(코딩)   ├→ 온폴리시 증류 → 단일 모델
                   → 뒤 단계가 앞을 잊음            └→ RL(에이전트)┘
```

- **DeepSeek-V4**(B): 도메인별로 SFT→GRPO 전문가를 **독립 육성** → **전어휘 KL** 온폴리시 증류로 통합. 토큰 단위 추정이 아니라 전어휘 손실을 쓰는 게 핵심 — 전문가들이 불일치할 때 그래디언트가 안정된다.
- **Kimi K3**(A): 3도메인 × 3추론강도 = **전문가 9개** → MOPD. 학생이 자기 롤아웃을 돌리고, 해당 도메인 교사가 **토큰별 밀집 보상**을 준다.

> **우리 Stage-2(일반) → Stage-3(의료) 순차 구조가 정확히 이 패턴이 대체하려는 것이다.**
> 그리고 우리는 이미 그 대가를 데이터로 봤다 — Stage-2 는 일반 +8.18pp 를 벌었지만
> **의료는 여섯 지점 전부 미검출**(p>0.5)이다. 순차 RL 이 목표 도메인으로 전이되지 않았다.

### ⑥ RL 알고리즘은 **수렴이 아니라 분화** 중이다

| 진영 | 대표 | 근거 |
|---|---|---|
| GRPO + 안정화 패치 | EXAONE 4.5(IcePop), HyperCLOVA X(KL 제거·비대칭 클리핑·동적 샘플링) | 메모리 효율 |
| GRPO 대체 | Qwen3-VL(**SAPO**) | 하드 클리핑의 정보 손실 |
| **PPO 회귀** | GLM-5.2 ⚠️C | 보상 해킹 억제에 critic 이 유리 |

HyperCLOVA X 32B Think 의 GRPO 개조가 특히 구체적이다(A):
- **KL 패널티 제거** — 생성 다양성 확보 목적
- **상단 클립 > 하단 클립** (비대칭)
- 동적 샘플링 + 적응 클리핑
- 정렬 단계는 PPO — "고정 시간 내 최적화 스텝 수가 GRPO 계열보다 많다"

> ⚠️ 우리 `beta=0.04`(KL 유지) 대 HyperCLOVA X 의 **KL 제거**는 정반대다. 다만 목적이 다르다 —
> 저쪽은 **다양성 확보**, 우리는 **정책 이탈 억제**. 우리 붕괴는 다양성 부족이 아니라 이탈이었으므로
> KL 을 없애는 방향은 위험하다. 비대칭 클리핑도 같은 이유로 신중해야 한다
> ([arXiv 2509.26114] clip-low 는 엔트로피를 **올리고** clip-high 는 **내린다**).

### ⑦ 불안정은 대부분 **아키텍처 층위**에서 잡는다

- **K-EXAONE 2.0**(A): 깊은 층 일부 전문가의 활성값 폭주 → **Clamped SwiGLU**(임계 7.0). 피크 활성 **6,862 → 48.96**.
- **Kimi K3**(A): latent MoE 라우팅 경로의 활성 폭발 → up-projection 앞 RMSNorm + Sigmoid-Tanh GLU. 극단 희소성(전문가 896개·56배)에서 auxiliary-loss-free 부하분산 붕괴 → **Quantile Balancing**.
- **Kimi K3**(A): **비전 타워 최적화 불안정** — SigLIP 초기화 인코더가 지속적으로 높은 gradient norm 과 잦은 스파이크를 보임 → **MoonViT-V2 를 next-token prediction 으로 스크래치 학습**해 해결.

> 마지막 항목이 우리와 관련 있다. 우리는 `freeze_vit=True` 로 비전 타워를 얼려 이 문제를 회피 중이다.
> 사후분석 §7 의 "`freeze_vit=False` 의료 가설"을 열기 전에 **Kimi 의 관찰을 기억할 것** — 비전
> 타워를 RL 에 열면 gradient norm 스파이크가 따라온다.

---

## 3. 콜드스타트 SFT 는 여전히 필요한가

### 결론: **필요하다. 다만 하는 일이 형식 학습이 아니다. 그리고 많이 하면 해롭다.**

**(a) 조사한 모델 중 순수 RL-from-base 는 하나도 없다.** 8개 전부 RL 앞에 SFT 가 있다.
Kimi K3 는 명시적으로 "SFT/**Cold-Start**" 라 부르고, 여기서 추정한 `b₀(x)` 를 **뒤 RL 단계의 길이 예산 기준선**으로 쓴다 — 콜드스타트가 형식뿐 아니라 **후속 RL 의 하이퍼파라미터를 공급**한다.

**(b) 그러나 "많을수록 좋다"가 2026년에 뒤집혔다.**

| 논문 | 발견 |
|---|---|
| [SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR](https://arxiv.org/pdf/2606.18487) | SFT 를 과하게 하면 출력 분포가 과집중(저엔트로피) → RLVR 신호가 안 먹힘. **SFT 후 순위와 RLVR 후 순위가 뒤집힌다**(rank inversion). 최적점이 존재하며 최대화가 아니다 |
| [The Synergy Dilemma of Long-CoT SFT and RL](https://arxiv.org/pdf/2507.07562) (추론 VLM 대상) | 과도한 long-CoT SFT 가 후속 RL 을 **방해**한다 |

**(c) 멀티모달에서는 콜드스타트의 *종류*가 결정적이다.** 🔴

[ReVisual-R1](https://github.com/CSfufu/Revisual-R1) (ICLR 2026) 은 3단 커리큘럼이다 — **텍스트 전용 콜드스타트 → 멀티모달 RL → 텍스트 전용 RL**. 그리고 관련 연구가 이유를 짚는다:

> **멀티모달 콜드스타트는 시각 주의 범위를 못 끌어올린다**(베이스 모델과 주의 분포가 거의 같다).
> **텍스트 전용 콜드스타트는 뚜렷이 올린다.** — "Lazy Attention Localization"

우리 Stage-1 은 멀티모달 혼합 콜드스타트(`sft_mixed_merged`)다. 이 발견이 맞다면 **우리 콜드스타트는 시각 주의를 못 키웠을 수 있다.** 의료(pmcvqa) 여섯 지점 전부 미검출인 것과 정합적인 가설이다 — 물론 확증은 아니다.

### 그래서 태그를 버리면?

**태그를 버려도 콜드스타트는 남는다.** 콜드스타트가 하는 일을 쪼개보면:

| 콜드스타트의 역할 | 태그를 버리면? |
|---|---|
| ① 출력 **형식** 학습 (`<think>…</think><answer>…</answer>`) | **사라짐** — 템플릿으로 내려가면 SFT 가 가르칠 게 없다 |
| ② **긴 사고사슬 행동** 자체 (태그와 무관) | **남음** — 베이스는 짧게 답한다. RL 만으로 long-CoT 를 세우는 건 훨씬 비싸다 |
| ③ RL 출발 정책 품질 = 탐색 부담 경감 | **남음** — 우리 실측: v2 0.185 → v3 0.909(형식), 홀드아웃 **0.295 → 0.348(+18%)**. 정확도가 같이 올랐다 |
| ④ 길이 예산 `b₀(x)` 등 **후속 RL 의 기준선 공급** | **남음** (Kimi K3 방식) |
| ⑤ 멀티모달: **시각 주의 유도** | **남음**, 단 텍스트 전용이 더 나을 수 있음 |

**①만 사라지고 ②③④⑤는 그대로다.** 우리 Stage-1 실측이 이를 뒷받침한다 — v3 콜드스타트는 형식만 고친 게 아니라 **홀드아웃 정확도를 +18% 올렸다.** 그건 태그와 무관한 이득이다.

---

## 4. 이 프로젝트에 대한 함의

### 처음부터 다시 돌린다면 — 우선순위

| | 조치 | 근거 등급 | 비용 |
|---|---|---|---|
| **1** | **정확도 보상을 형식에서 분리** (완료, `2a0458a`) | 자체 실측 + ①패턴 | 0 |
| **2** | **형식 태그를 채팅 템플릿으로 내리기.** 보상의 format 항은 언어일관성·반복억제와 같은 급의 보조항으로 축소 | A (HyperCLOVA X) | 낮음 |
| **3** | **길이를 하드 예산으로.** `b₀(x)` 추정 후 σ 배 초과 시 보상 −1 | A (Kimi K3) | 낮음 |
| **4** | **학습–추론 logprob 불일치 계측 추가.** 5% 임계 감시 | A (IcePop) | 낮음 |
| **5** | `scale_rewards` 나눗셈 제거 + zero-variance 그룹 필터 | A (EXAONE 4.5) | 0 |
| **6** | `overlong_filter=False` | 자체 실측 (사후분석 §4-4) | 0 |
| **7** | 콜드스타트를 **텍스트 전용**으로 바꿔 시각 주의 확인 | B (ReVisual-R1) | 중간 — Stage-1 재실행 |
| **8** | **Stage-2/3 를 순차에서 병렬 전문가+증류로** 재구성 | B (DeepSeek-V4, Kimi K3) | **높음 — 구조 변경** |

1~6 은 다음 실행에 바로 넣을 수 있고 GPU 추가 비용이 거의 없다. 7~8 은 설계 변경이라 별도 판단이 필요하다.

### 8번을 진지하게 볼 이유

우리 데이터가 이미 순차 구조의 한계를 보여준다 — Stage-2 는 일반 +8.18pp, **의료 +0.75pp(p>0.5, n=400)**. 계획서의 산출물은 의료다. 2026년의 두 프런티어 랩이 독립적으로 **"도메인별로 따로 키우고 증류로 합친다"**에 도달했다는 건, 순차 RL 의 도메인 전이 실패가 우리만의 문제가 아니라는 뜻이다.

다만 우리 하드웨어(A100 80GB **PCIe**, NVLink 없음)에서 전어휘 KL 증류는 통신 비용이 크다. **작게 시작하는 길** — 전문가를 2개(일반·의료)만 두고, 증류 대신 **가중 평균 병합**부터 시험해 보는 것.

---

## 5. 출처

**테크니컬 리포트 (A/B 등급)**
- [K-EXAONE 2.0 Technical Report (arXiv 2608.04505)](https://arxiv.org/abs/2608.04505) · [K-EXAONE (arXiv 2601.01739)](https://arxiv.org/abs/2601.01739)
- [EXAONE 4.5 Technical Report (arXiv 2604.08644)](https://arxiv.org/abs/2604.08644)
- [Kimi K3: Open Frontier Intelligence (arXiv 2607.24653)](https://arxiv.org/abs/2607.24653)
- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (arXiv 2606.19348)](https://arxiv.org/html/2606.19348v1)
- [Qwen3.5-Omni Technical Report (arXiv 2604.15804)](https://arxiv.org/html/2604.15804v1)
- [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/abs/2511.21631)
- [HyperCLOVA X 32B Think (arXiv 2601.03286)](https://arxiv.org/html/2601.03286v1) · [HyperCLOVA X THINK (arXiv 2506.22403)](https://arxiv.org/abs/2506.22403)
- [Solar Open Technical Report (arXiv 2601.07022)](https://arxiv.org/pdf/2601.07022)

**알고리즘·안정성**
- [Soft Adaptive Policy Optimization — SAPO (arXiv 2511.20347)](https://arxiv.org/html/2511.20347v1)
- [IcePop / Ring-flash-2.0 — MoE RL 안정화](https://ant-ling.medium.com/ring-flash-2-0-4bf5b62204f5) · [Every Step Evolves: Trillion-Scale Thinking Model (arXiv 2510.18855)](https://arxiv.org/pdf/2510.18855)
- [Stabilizing MoE RL by Aligning Training and Inference Routers (arXiv 2510.11370)](https://arxiv.org/pdf/2510.11370)
- [Clip-Low Increases Entropy and Clip-High Decreases Entropy (arXiv 2509.26114)](https://arxiv.org/pdf/2509.26114)
- [Understanding and Preventing Entropy Collapse in RLVR (arXiv 2605.11491)](https://arxiv.org/html/2605.11491v1)

**콜드스타트 SFT 의 양·종류**
- [SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR (arXiv 2606.18487)](https://arxiv.org/pdf/2606.18487)
- [The Synergy Dilemma of Long-CoT SFT and RL (arXiv 2507.07562)](https://arxiv.org/pdf/2507.07562)
- [ReVisual-R1 (ICLR 2026)](https://github.com/CSfufu/Revisual-R1) · [From Narrow to Panoramic Vision: Attention-Guided Cold-Start (arXiv 2603.03825)](https://arxiv.org/html/2603.03825v1)
- [Advancing Multimodal Reasoning: Optimized Cold Start to Staged RL (arXiv 2506.04207)](https://arxiv.org/pdf/2506.04207)
- [Skywork-R1V3 Technical Report (arXiv 2507.06167)](https://arxiv.org/pdf/2507.06167)

**2차 출처 (C 등급 — 방향성 참고용)**
- [GLM-5.2 정리 (Medium, 2026-06)](https://machine-learning-made-simple.medium.com/understanding-glm-5-2-beyond-the-headlines-3a4e654c9542) · [Zhipu GLM-5.2 분석](https://weijinresearch.substack.com/p/zhipus-glm-52-a-usability-breakthrough)
- [Kimi K3 vs DeepSeek V4 vs GLM-5.2 비교](https://deepinfra.com/blog/kimi-k3-vs-deepseek-v4-pro-vs-glm-5-2) · [Chinese LLMs 2026](https://www.turingpost.com/p/llms-in-china)
- [DeepSeek V4: ten teachers, one student](https://maximelabonne.substack.com/p/deepseek-v4-ten-teachers-one-student)
