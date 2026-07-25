# RLVR 하이퍼파라미터 — 외부 관행 대조 (2026 리포트 조사)

> 목적: 이 프로젝트의 Stage-2 RLVR(GRPO/GDPO) 설정을 **외부에 알려진 관행**과 대조하고,
> 자원 제약으로 벌어진 지점과 실험 우선순위를 명시한다.
> 프로젝트 실측 근거는 [`../README.md`](../README.md)(Stage-2 · step600 포화)·[`../HANDOFF.md`](../HANDOFF.md),
> 실행 절차는 [`stage2_expansion_runbook.md`](stage2_expansion_runbook.md).

## 0. 조사 프레이밍 (솔직한 한계)

- **2026년 프런티어 리포트**(Qwen3.6·DeepSeek V4·Kimi K2.6·MiniMax M2 등)는 아키텍처는
  자세히 공개해도 **RL 하이퍼파라미터 전체 표는 거의 공개하지 않는다.**
- 구체적 RLVR 레시피가 공개된 것은 주로 **2025년 중반 리포트**(Magistral, MiniMax-M1, DAPO, Dr.GRPO).
- **2026년의 기여**는 대부분 "**얼마나 오래 학습할지**"(과학습·다양성 붕괴)를 정밀화한 연구 논문.
- 아래는 그 둘을 합친 **현재 통용 관행**이며, 절대 정답이 아니라 **다수설 + 근거 출처**다.

## 1. 하이퍼파라미터 대조표

| 파라미터 | 외부 관행(다수설) | 근거 | 이 프로젝트 | 판정 |
|---|---|---|---|---|
| **KL 페널티 β** | **0으로 제거**가 대세 | Magistral "remove KL entirely" · MiniMax-M1(CISPO) "no KL penalty" · DAPO · Open-Reasoner-Zero("removing KL→optimal") | **0.04** | ⚠️ 보수적. 검증가능 보상엔 KL 불필요/유해가 다수설 |
| **학습률(actor)** | 작고 **상수**, full-FT ~**1e-6** | DAPO/Dr.GRPO 계열; 스윕해도 marginal | LoRA **1e-5** | ✅ LoRA라 더 높은 게 정상 |
| **그룹 크기(rollouts/prompt)** | **8~16** | DAPO/NeMo=16 · 다수=8 · "8→4는 정확도 하락" | **4** | ⚠️ 관행 하한 아래(메모리 절충) |
| **배치(prompt/step)** | 매우 큼(512~수천 시퀀스) | DAPO/NeMo=512×16 · Magistral 2k~8k | **32**(1×accum4×8gpu) | ⚠️ 소규모(8GPU LoRA 제약) |
| **clip ε** | εlow 0.2 / **εhigh 0.26~0.30**(clip-higher) | DAPO 0.28 · Magistral 0.26~0.3 | 0.2 대칭 | ✅ dr_grpo는 clip-higher 미적용이 정석 |
| **손실 정규화** | 길이편향 제거(토큰레벨/상수) | Dr.GRPO · DAPO · Magistral 공통 | dr_grpo loss | ✅ 일치 |
| **advantage 스케일** | std 나눗셈 제거 or minibatch 정규화 | Dr.GRPO `scale_rewards=none` · Magistral minibatch | GDPO(보상별 정규화) | ✅ 최신 계열 |
| **dynamic sampling / overlong filter** | 표준(acc 0·1 그룹 폐기, 잘림 제외) | DAPO | 둘 다 有 | ✅ |
| **롤아웃 temperature** | **~1.0**(탐색) | ORZ "T=1 best" · Magistral 0.7~1.0 | **0.9**(ms-swift 기본) | 🔧 문헌 1.0과 미세차, A/B 가치 |

**요약**: 손실·advantage·필터링(코어 알고리즘)은 최신 관행과 **정합**. 관행과 벌어진 4곳
(KL β · 그룹 4 · 배치 32 · temp 0.9)은 **전부 8GPU·LoRA·no-NVLink 자원 제약**의 결과이며,
치명적 오류가 아니라 "탐색 신호가 다소 약할 수 있는" 방향의 트레이드오프다.

## 2. 에포크 — "많이"가 아니라 "1~수 에포크 + 조기중단"

RLVR은 SFT와 달리 **에포크를 많이 돌리면 안 된다.** 프로젝트 실측(step600 포화)과 외부 근거가 일치.

- **《Understanding Diversity Collapse in RLVR via Overtraining》(arXiv 2606.15455, 2026-06)**
  - MATH/Qwen2.5-Math-7B: **검증 Pass@1은 ~5에포크에서 정체**. 이후 train 성공률만 68.7%→77.1%,
    검증은 ~72% 평평.
  - "**후반 에포크는 Pass@1 이득 거의 없이 high-k Pass@k(다양성)만 깎는다**" → **5에포크 조기중단 권고**.
  - 표준 n=8 세팅에선 "**대부분의 업데이트가 추론경계엔 과학습**".
  - 기전: 문제 기여도 포화 → 이후 업데이트는 능력 확장이 아니라 **선호 궤적에 확률질량 집중
    (policy sharpening) → 다양성 붕괴**.
- **프런티어 리포트는 에포크로 세지 않는다**: Magistral·MiniMax-M1은 대형 배치 + 신선한 롤아웃 +
  **길이 점증**(16k→24k→32k)으로 **스텝 단위**로 돌리고 평가곡선으로 멈춘다.

**환산(이 프로젝트)**: 확장셋 74,787 ÷ 32 = **1 epoch ≈ 2,337 step**.
step600 = 19,200 prompt ≈ **0.26 epoch** → 1에포크도 못 돌고 이미 포화.

**결론**: 에포크 상한을 늘리지 말 것. **≤1에포크, 50스텝마다 홀드아웃 평가, 포화 시 조기중단**이 정답.

## 3. 실행 제언 (우선순위)

| # | 제언 | 방법 | 우선순위 |
|---|---|---|---|
| 1 | **그룹 4 → 8** 재고 | `NUM_GEN=8 bash scripts/launch_stage2_expanded.sh` (길이예산·메모리 트레이드오프 측정) | 중 |
| 2 | **롤아웃 temp 0.9 → 1.0** A/B | `TEMPERATURE=1.0 …` (문헌 다수설) | 중 |
| 3 | **KL β 0.04 → 0.01** 탐색 여지 확대 | `BETA=0.01 …` (초기 붕괴 없으면) | 낮 |
| 4 | **에포크 상한 유지** — 늘리지 말 것 | 지금 계획(≤1ep·조기중단) 그대로 | — |

> ⚠️ **재현성 원칙**: 위는 전부 **env override 로 실험**하며, 스크립트 **기본값은 검증된 A/B 값
> (NUM_GEN 4 · BETA 0.04 · TEMPERATURE 0.9) 그대로 둔다.** 기본값을 바꾸면 과거 A/B와 clean 비교가 깨진다.
> 각 knob 은 `scripts/21_rlvr_grpo_adv.slurm` 상단 주석 및 `launch_stage2_expanded.sh` 사용법에 노출돼 있다.

## 4. 출처

- Understanding Diversity Collapse in RLVR via the Lens of Overtraining — arXiv:2606.15455
  <https://arxiv.org/html/2606.15455v1>
- Magistral (Mistral) technical report — arXiv:2506.10910 <https://arxiv.org/html/2506.10910v1>
  (KL 제거·advantage minibatch 정규화·길이정규화·εhigh 0.26~0.3·temp 0.7~1.0·길이 16k→32k 점증)
- MiniMax-M1: Scaling Test-Time Compute — arXiv:2506.13585 <https://arxiv.org/html/2506.13585v1>
  (CISPO=importance weight 클리핑, KL 제거, temp 1.0/top-p 0.95, 40k/80k 길이)
- DAPO walkthrough — NVIDIA NeMo-RL <https://docs.nvidia.com/nemo/rl/latest/guides/dapo.html>
  (num_prompts_per_step 512 · num_generations 16 · εhigh 0.28 · overlong_buffer 4096 · max_resp 20480)
- Interconnects: base model RL & GRPO tweaks (Open-Reasoner-Zero, Dr.GRPO)
  <https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo>
- DAPO explainer <https://medium.com/@syed_hasan/dapo-decoupled-clip-and-dynamic-sampling-policy-optimization-grpo-on-steroids-9c571a0536f3>
- GRPO fine-tuning 2026 practical guide (Spheron) <https://www.spheron.network/blog/grpo-fine-tuning-gpu-cloud/>

_조사 2026-07-24. ms-swift 4.13 기준 rollout temperature 기본값 = 0.9 (`swift/arguments/rlhf_args.py`)._
