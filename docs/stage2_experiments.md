# Stage-2 · 범용 RLVR (GRPO) — 실험 상세

> 이 문서는 [`README.md`](../README.md) 에서 분리된 상세 기록입니다. 요약·현황은 README, 상세는 여기.

## Stage-2 · 범용 RLVR (GRPO)

> **요약 (Stage-2 방법론 실험 완료)**: baseline GRPO 에서 **Acc plateau** 진단 → GRPO 파생기법 5종 **clean A/B** → **dr_grpo 승자**(plateau 돌파, 홀드아웃 +73%). 최신기법 **GSPO·GDPO**도 검증: **GDPO 홀드아웃 동률**(0.390 vs 0.380, Stage-3용 채택 권고), **GSPO는 on-policy 동률이나 홀드아웃 열위**(0.290=일반화 실패, 미채택). step600 홀드아웃 **~0.38–0.39 포화**. **Stage-2 방법론 확정**.

### 0) 풀확장 재설계 (2026-07-22~24 데이터 · 07-28 본실행 착수)

**동기**: v3 콜드스타트 A/B 에서 **math 가 v2 와 동률(0.3245)** — vl 은 올렸으나 수치계산은 못 올림. 게다가 프로젝트 목표는 **의료**인데 Stage-2 가 일반 전용이었음. → **DeepVision 단일 → math(MMK12) + 의료(PMC-VQA) 추가**. (예산: 2026-07-27 **k252a02 로 이관 완료** — 신규 5,000 노드시간에서 2,337 step ≈1,674~1,719(33~34%) 집행, 나머지는 Stage-3 유보.)

**확장셋 (의료 27% 확정, 2026-07-24 재조립):**

| 소스 | 설명 | 학습 | 홀드아웃 | 정답형식 | 역할 |
|---|---|---|---|---|---|
| DeepVision-103K | 수학·시각논리 멀티모달 문제(일반, 검증가능 정답) | 40,000 | 972 | math/MC | 일반 시각추론 base(RL 검증됨) |
| **MMK12** | 중국 K-12 수학 시험문제(**영어 번역본**, 이미지+수치/수식 답) | 15,204 | 400 | 수치/수식 82% | **순수 math** — 약점 직격 |
| **PMC-VQA** | PubMed 논문 그림 기반 의료 VQA(4지선다) | 19,583 | 400 | **MC(B/C/A/D 균형)** | **의료 광범위** — 목표 정렬 |
| **합계** | — | **74,787** | **1,772** | | 일반53 / math20 / 의료26 |

- **데이터 = "받아서 실측"으로 선별**: Kvasir(58K인데 고유이미지 671·degenerate)·SLAKE(이미지 450)·ThinkLite(노이즈)는 실측 후 탈락. **PMC-VQA만 대형·다양·검증가능 의료**(329K MC·PubMed 광범위). 상세 → [`docs/stage2_data.md`](stage2_data.md).
- **오염 최종 해소**: 전 소스 **이미지 바이트해시 dedup**. 이번 재조립에서 **DeepVision 구 22% 오염까지 제거**(홀드아웃 이미지해시를 train 서브샘플서 배제) → 전 소스 홀드아웃 1,772건 누수 0 검증.
- **init = v3** `sft_mixed_merged` · **레시피 = GDPO**(`21_rlvr_grpo_adv` 경유, dynamic_sample 코어 포함). 실행 → [`docs/stage2_expansion_runbook.md`](stage2_expansion_runbook.md).
- **하이퍼파라미터 외부 관행 대조** → [`docs/rlvr_hparams_external.md`](rlvr_hparams_external.md): 2026 리포트 기준 KL β·그룹크기·temp·에포크 관행과 대조. 코어(손실·advantage·필터)는 관행 정합, 벌어진 4곳(β 0.04·그룹 4·배치 32·temp 0.9)은 자원 제약. **에포크는 ≤1·조기중단이 정설**. 기본값은 검증값 `NUM_GEN=4`·`TEMPERATURE=0.9`·`BETA=0.04` 유지, A/B override 후보가 `NUM_GEN=8`/`TEMPERATURE=1.0`/`BETA=0.01`.
- 스크립트: `build_pmcvqa.py`·`13_build_stage2_expanded.slurm`(변환) · `build_stage2_mix.py`(조립·`DV_CAP`/`PMC_CAP` 비율조정) · `launch_stage2_expanded.sh`(GDPO 제출).
- 🚀 **본실행 진행중(2026-07-28~)**: 배선 스모크 완주(1GPU job 72844 · 8GPU job 72832, 5/5 step, `frac_reward_zero_std` 0) → **2,337 step 체인 제출**(`scripts/launch_stage2_expanded_epoch.sh`, afterany 4잡, ~209h). 재제출 후 현재 체인은 **job 73924~73927**(73312~73315 는 구 체인).
  - **학습량 결정**: 구 데이터셋에선 step600 포화였으나 **확장셋은 MMK12·PMC-VQA 가 새로 들어가 포화점이 다를 수 있어** 학습량을 늘렸다. `MAX_STEPS` 를 처음부터 목표치로 지정해야 LR 이 매끄럽게 감쇠(600→2337 로 늘리면 불연속).
  - ⚠️ **2026-08-02 정정**: 이 2,337 step 을 "1 epoch" 으로 적어 왔으나 실제로는 **0.25 epoch** 이다(1 epoch = 74,787 ÷ 8 = 9,348 step). 확장셋의 약 75%는 미노출로 남는다. → [`stage2_run73924_progress.md`](stage2_run73924_progress.md) §3
  - 📊 **중간 점검(step 630)**: 인프라 무결(오류 0건)이나 **AccuracyMix 가 630 step 동안 +0.7% 로 정지**, 대신 completion 길이 +29.8%·클리핑 +47.5%. → [`stage2_run73924_progress.md`](stage2_run73924_progress.md)
  - ⚠️ **조기중단 원칙**: 외부 문헌의 diversity collapse 경고(후반 구간은 Pass@1 이득 없이 high-k Pass@k 감소)에 따라 **중간 체크포인트(save_steps 50)를 소스별(`_source`) 홀드아웃으로 평가하고 포화 시 중단**. → [`docs/rlvr_hparams_external.md`](rlvr_hparams_external.md)

### 1) baseline → Acc plateau 진단 (job 57249, step 1000 완주)

RFT 콜드스타트 init + LoRA-DDP + max_completion 6144. 100-step 구간평균:

| 구간 | reward | Acc | FormatThink | clip | mean_len | zero_std |
|------|--------|-----|-------------|------|----------|----------|
| 1–100 | 0.377 | 0.418 | 0.259 | 0.420 | 3778 | 0.240 |
| 501–600 | 0.534 | **0.500** | 0.525 | 0.307 | 3309 | 0.249 |
| 901–1000 | **0.557** | 0.491 | **0.660** | **0.279** | **3267** | **0.328** |

- ✅ 발산/붕괴 없음. FormatThink +155%·clip↓·길이↓.
- ⚠️ **[진단] Acc plateau**: Acc가 step~500서 0.50 정점 후 정체. 원인 = **`frac_reward_zero_std` 0.24→0.33 상승**(그룹 rollout이 전부정답/전부오답 → 정확도 gradient 소실). 후반 reward 상승은 형식·길이 주도.

### 2) 기법 통합 비교 — GRPO 계열 5종 clean A/B

baseline plateau를 뚫기 위해 GRPO 파생기법 5종을 **기법 외 전 조건 동일**(init `sft_rft_coldstart_merged` · trainonly 데이터 · 보상 `accuracy_mix/format_think/soft_overlong` · LoRA r16/a32)로 실증. 전부 **한 스크립트**(`scripts/21_rlvr_grpo_adv.slurm`)에서 `RECIPE`/env 토글로 실행 → 진짜 clean A/B.

#### 쉬운 설명 (비유)

> **비유**: 한 문제에 모델이 **답안 4장(그룹)**을 낸다 → 정답 여부만 채점 → **그룹 평균보다 잘 쓴 답은 "더 그렇게", 못 쓴 답은 "덜 그렇게"** 확률을 민다.
> PPO와 달리 **별도 채점관(가치망) 불필요** — *그룹 평균*이 기준선 역할(=GRPO의 핵심 절약).

5종 모두 이 **그룹상대 advantage** 골격을 공유하고, "채점을 점수로 환산하는 규칙"에서만 갈린다.

#### 핵심 관점: 5종 = ms-swift **3개 독립 손잡이(knob)**의 조합

방법을 통째로 외우지 말고, ms-swift가 노출하는 **직교하는 3개 손잡이**의 좌표로 본다 (소스: `grpo_trainer.py` 손실 분기 직접 확인). 손잡이마다 겨냥하는 편향이 다르다:

| 손잡이 (ms-swift 인자) | 무엇을 바꾸나 | 선택지와 효과 |
|------|------|------|
| **`loss_type`** | 손실 정규화 + 클리핑 | `grpo`=÷시퀀스길이(**길이 편향**) · `dapo`=÷배치토큰+clip-higher(탐색↑) · `dr_grpo`=÷상수(**길이 편향 제거**) |
| **`importance_sampling_level`** | 중요도샘플링(IS) 단위 | `token`=기본 · `sequence`=**GSPO**(장문 IS 분산·클립증폭 억제) |
| **`scale_rewards`** | advantage 정규화 | `group`=÷그룹std(**난이도 편향**) · `none`=안 함(**dr_grpo**) · `gdpo`=**보상함수별 개별** z-score(멀티리워드 균형) |

→ 세 손잡이는 **독립(자유 조합 가능)**. 승자 **dr_grpo = `(dr_grpo, token, none)`**. **GDPO**는 거기서 `scale_rewards`만 `gdpo`로 돌린 **1-knob 변형**(`dr_grpo, token, gdpo`). **GSPO**는 IS를 sequence로 바꾼 **별도 좌표**(우리 설정 `grpo, sequence, group` + 작은 clip)로, dr_grpo의 1-knob 변형이 아니라 독립 실험이다.

#### 5종 실제 설정·결과 통합표

공통 **CORE**(baseline 제외 전 recipe 공유) = `dynamic_sample`(zero_std 그룹 폐기·재샘플, plateau 직격) + `overlong_filter`(잘린 롤아웃 loss 제외).

| 방법 | 논문 | `loss_type` | IS | `scale_rewards` | 클립 ε/ε_high | CORE | **우리 결과·판정** |
|------|------|------|------|------|------|:---:|------|
| **GRPO** (baseline) | [2402.03300](https://arxiv.org/abs/2402.03300) | grpo | token | group | 0.2 | ✗ | plateau (zero_std 0.24→0.33), Acc 0.500 정점 후 정체 |
| **DAPO** | [2503.14476](https://arxiv.org/abs/2503.14476) | dapo | token | group | **0.2/0.28** | ✓ | 안정(zero_std 0)하나 **미돌파**(Acc 0.465<0.500), 길이 재폭주(~3600) |
| **dr_grpo** ✅ | [2503.20783](https://arxiv.org/abs/2503.20783) | **dr_grpo** | token | **none** | 0.2 | ✓ | ✅ **돌파·승자**(Acc **0.526**·길이 억제 3259), **홀드아웃 +73% 검증** |
| **GSPO** | [2507.18071](https://arxiv.org/abs/2507.18071) | grpo | **sequence** | group | **3e-4/4e-4** | ✓ | **미채택**: on-policy 동률(0.500)이나 **홀드아웃 열위 0.290**(dr 0.38·GDPO 0.39 대비)=일반화 실패 |
| **GDPO** | [2601.05242](https://arxiv.org/abs/2601.05242) | dr_grpo | token | **gdpo** | 0.2 | ✓ | **동률**(step600 홀드아웃 0.390 vs dr 0.380, Δ+0.01 노이즈 내) → Stage-2 무차별, **Stage-3용 채택 권고** |

**판정 핵심**: plateau의 원인은 `frac_reward_zero_std` 급등(그룹이 전부정답/전부오답→gradient 소실). dr_grpo가 **길이·난이도 두 정규화 편향을 동시에 제거**해 step 501~600서 Acc 0.50을 유일하게 돌파. GSPO·GDPO는 그 위/옆의 직교 개선을 clean A/B로 검증했고, **둘 다 dr_grpo와 동률**(GSPO 미채택, GDPO는 멀티리워드 균형이 필요한 Stage-3용으로 채택 권고 — [아래 판정](#stage-2-3종-홀드아웃-판정-step600-n200)).

기법마다 학습 데이터가 달라(구 데이터 vs trainonly) **비교 가능한 2개 코호트**로 나눠 그림(각 6패널: Acc·reward·FormatThink·mean_len·zero_std·clip, 50-step 구간평균, 노란 띠=판정창 501~600):

**코호트 A — plateau 돌파 A/B (구 데이터): baseline vs DAPO vs dr_grpo**
![Stage-2 A: baseline/DAPO/dr_grpo](assets/grpo_stage2_A_plateau.png)
*(**dr_grpo(초록)만 노란 띠서 Acc 0.50 돌파**하며 mean_length 억제. baseline(파랑)은 zero_std 상승=plateau, DAPO(주황)는 길이·clip 재증가. FormatThink는 baseline이 최고지만 그건 Acc 정체를 형식이 대신 끌어올린 것.)*

**코호트 B — 최신기법 A/B (trainonly, 동일 init·데이터): dr_grpo(none) vs GSPO vs GDPO**
![Stage-2 B: dr_grpo/GSPO/GDPO](assets/grpo_stage2_B_latest.png)
*(3기법 모두 step600 완주. Acc·reward는 3자 거의 겹침(**동률**). FormatThink는 GDPO(보라)가 중반 우위였으나 **판정창(노란 띠)서 dr_grpo와 수렴**. **clip에서 GSPO(빨강)만 크게 높음**(~0.18, 작은 ε 설계상). → GSPO·GDPO 둘 다 dr_grpo와 동률: GSPO 미채택, GDPO는 Stage-3용 채택 권고.)*

<details><summary>플롯 재생성 명령</summary>

```bash
SB=work/images/ms-swift-413-sandbox
# 코호트 A
singularity exec $SB python scripts/plot_grpo_multi.py docs/assets/grpo_stage2_A_plateau.png \
  "baseline:#1f77b4:-:logs/grpo_stage2_57249.log" "DAPO:#ff7f0e:--:logs/grpo_adv_57527.log" "dr_grpo:#2ca02c:-.:logs/grpo_adv_57624.log"
# 코호트 B
singularity exec $SB python scripts/plot_grpo_multi.py docs/assets/grpo_stage2_B_latest.png \
  "dr_grpo(none):#2ca02c:-.:logs/grpo_adv_58892.log" "GSPO:#d62728:--:logs/grpo_adv_59004.log" "GDPO:#9467bd:-:logs/grpo_adv_59191.log"
```
`plot_grpo_multi.py` 는 `label:color:style:logpath` 스펙을 N개 받아 일반화(기법 수 무관).
</details>

> 원칙: **"최신이라서" 채택하지 않는다.** 5종 전부 dr_grpo가 자리를 얻은 것과 **동일 clean A/B**(동일 init·trainonly, 판정창 501~600 구간평균 + 필요시 층화 홀드아웃 벤치마크)로 판정. 동률이면 이미 검증된 승자를 유지.

<details><summary>① GRPO / DAPO / dr.GRPO — 한 줄 요약 + DAPO 4대 기법</summary>

- **GRPO** (DeepSeekMath, 2024) = 기본형. 그룹 평균 빼고 **그룹 std로 나눠** 점수화. 약점: ① std=0 그룹은 못 배움(**Acc 정체**) ② 긴 답·쉬운 문제 과대평가.
- **DAPO** (ByteDance, 2025) = GRPO에 **탐색·효율 4종 보강**. 안정성↑이나 답이 길어져 정확도 baseline 미달.
- **dr.GRPO** (2025, "R1-Zero 비판적 분석") = GRPO의 **편향을 수술**(손실 상수정규화 + std나눗셈 삭제). DAPO의 "답 길어짐"을 정면 겨냥. **우리 승자**.

| DAPO 기법 | ms-swift 인자 | 효과 | plateau 관련성 |
|------|--------------|------|---------------|
| **Dynamic Sampling** | `--dynamic_sample true --max_resample_times 3` | reward_std=0 그룹 폐기·재샘플 → 유효 gradient↑ | ⭐ **직격** |
| **Clip-Higher** | `--epsilon 0.2 --epsilon_high 0.28` | 상단 클립 완화 → 저확률 토큰 탐색 보존 | 조기수렴/다양성소실 방지 |
| **Token-level Loss** | `--loss_type dapo` | 토큰 단위 정규화 → 길이 정규화 편향 제거 | 형식·길이 주도 reward 보정 |
| **Overlong handling** | `--overlong_filter true` (+`soft_overlong`) | 잘린 롤아웃 loss 제외 | 잘림=0점 신호 오염 제거 |

- 비용: DAPO 본실행(57527) 실측 **~369s/it (baseline 202의 ~1.8배)** — dynamic_sample 재샘플 대가.
- 원논문 성과: GRPO 수학추론 SOTA(7B) · DAPO AIME24 50(32B) · dr.GRPO AIME24 43.3%(7B, 토큰효율↑).
</details>

<details><summary>② GSPO (시퀀스 IS) — 배경·A/B 결론</summary>

**GSPO**(Group Sequence Policy Optimization, Alibaba/Qwen, 2025-07): 토큰 단위 IS를 **시퀀스(답안) 단위**로 교체.
- **왜**: 토큰 IS는 응답이 길수록 **분산 노이즈 누적** + 클립 증폭 → 붕괴(특히 MoE·장문·LoRA). 시퀀스 우도 비율(길이 정규화)로 억제 → MoE RL 안정화가 대표 성과.
- **A/B 결론(2026-07-04, 판정창 501~600, dr_grpo n=100 vs GSPO n=99)**: **사실상 동률**(Acc 0.500 vs 0.487, reward 0.519 vs 0.506, 차이 ±0.013 노이즈). GSPO 클립 훨씬 큼(0.185 vs ~0.001, 작은 ε 설계상).
- **→ dr_grpo 유지 (미채택)**: 동률 + dr_grpo는 이미 **홀드아웃 +73% 검증**됐고 GSPO는 on-policy 동률일 뿐. 시퀀스 IS가 필요한 MoE·초장문 상황도 아님. 런처 `scripts/launch_gspo_ab.sh`(`job 59004`).
</details>

<details><summary>③ GDPO (멀티리워드 정규화) — 배경·중간추세</summary>

**GDPO**(Group reward-Decoupled Normalization, NVIDIA, 2026-01): 여러 보상을 가중합→통짜 정규화하면 서로 다른 조합이 같은 advantage로 **뭉개지는(collapse)** 문제를, **보상 함수별 개별 정규화(z-score) 후 결합**으로 해결(AIME서 GRPO 대비 +2.3~6.3%).
- **왜 우리에게**: 우리는 **다중 보상**(Stage-2 3종, Stage-3 clinical_judge/format)이라 스케일이 제각각 → GDPO 타깃과 정확히 일치. 가장 유망한 적용처는 스케일차 큰 **Stage-3(judge 1.0 + format 0.2)**.
- **dr_grpo와 관계**: `scale_rewards`만 `none→gdpo`로 바꾼 **1변수 A/B**(나머지 dr_grpo 처리 전부 유지). ⚠️ dr_grpo가 없앤 ÷std가 **보상함수별로** 되살아남(목적은 난이도편향 회피가 아니라 멀티리워드 균형). swift 제약: `kl_in_reward=True` 비호환(우린 KL을 loss에 넣어 무관).
- **최종 판정**: step600 완주 → on-policy 판정창(501~600) Acc **0.487 vs 0.490 동률**, 층화 홀드아웃(N=200) **0.380 vs 0.390 동률**(GDPO 미세우위, 노이즈 내) → [상세 판정표](#stage-2-3종-홀드아웃-판정-step600-n200). Stage-3용 채택 권고.
- **학습 중 추세(`job 59191`, dr_grpo와 동일구간, dr/gd 순)**:

  | 구간 | reward | AccMix | FormatThink |
  |------|--------|--------|-------------|
  | 151–200 | 0.530/0.528 | 0.514/0.506 | 0.428/**0.452** |
  | 201–250 | 0.503/0.513 | 0.484/0.482 | 0.434/**0.493** |
  | 251–300 | 0.502/0.501 | 0.483/0.470 | 0.455/**0.498** |
  | 301–350 | 0.506/0.521 | 0.481/0.486 | 0.465/**0.519** |
  | 351–400 | 0.507/0.491 | 0.482/0.465 | 0.495/0.498 |
  | 401–450 | 0.530/0.535 | 0.501/0.497 | **0.520**/0.484 |

  - **AccMix·총 reward: 전 구간 dr_grpo와 동급**(Δ ±0.01~0.02) — 정답률 손해 없음. `zero_std=0` 유지.
  - **FormatThink: 150~350스텝 GDPO 우위**(+0.04~0.06)였으나 **400대에서 dr_grpo와 근접 수렴**(엎치락뒤치락) — 초반만큼 격차가 확실하진 않음. → 최종 판정이 더 중요해짐.
  - on-policy 지표라 **최종 판정은 step 501~600 + 층화 홀드아웃 벤치마크**로 확정(진입 임박).
</details>

### 3) base vs 학습모델 벤치마크 (층화 홀드아웃, 정식)

**중간 벤치마크(2026-07-03, RL 25%=step 800)** — 층화 홀드아웃 N=100, math/vl 층별(`eval_midtrain.slurm`). 조기 판단용으로 base / init(RL 0%) / trained(중간) 3개 동일조건 채점:

| 모델 | **전체 Acc** | math | visual-logic | 형식(`<answer>`) | 평균길이 |
|------|------|------|------|------|------|
| **base** (Qwen3.5-9B) | 0.15 | 0.192 | 0.113 | 0.12 | 5907자 |
| **init** (SFT 콜드스타트, RL 0%) | 0.22 | 0.234 | 0.208 | 0.32 | 5188자 |
| **trained** (dr_grpo step 800, **RL 25%**) | **0.38** | 0.340 | **0.415** | 0.45 | 4752자 |

- 🎯 **init → trained: 0.22 → 0.38 (+0.16, 상대 +73%)** — **Stage-2 RL 이 홀드아웃 정확도를 확실히 개선**(25%만 학습했는데도). *(당시엔 전량 학습 근거였으나, 이후 step600 재측정서 포화 확인 → 아래 참조)*
- base → trained **+153%**(전체 파이프라인 SFT+RL). visual-logic 개선 최대(0.21→0.42, +100%), 형식 0.12→0.45, 길이 5907→4752자(간결화).
- ⚠️ **"학습 롤아웃 Acc ~0.50 정체"는 오해였음**: on-policy(탐색 포함) 지표라 평탄했을 뿐, 정작 중요한 **홀드아웃에선 뚜렷이 상승**.
- ✅ **step600 재측정으로 확정**(아래 GDPO 판정표): dr_grpo/GDPO 둘 다 **0.38–0.39** → step800(0.38)과 동일 = **홀드아웃이 step600쯤 포화**(전량 학습해도 큰 추가이득 없음, Stage-2 조기확정 근거).

#### Stage-2 3종 홀드아웃 판정 (step600, N=200)

trainonly·콜드스타트 init·step600으로 **완전 동일 조건** 학습된 3종(dr_grpo/GDPO/GSPO)의 체크포인트를 병합해 **같은 층화 홀드아웃(N=200)**에서 대조 (`46_eval_gdpo_ab.slurm`·`48_eval_gspo_holdout.slurm`):

| 모델 (step600) | 홀드아웃 Acc | math | visual-logic | 형식 | (참고) on-policy |
|------|------|------|------|------|------|
| **GDPO**(gdpo) | **0.390** | 0.379 | 0.400 | 0.445 | 0.490 |
| **dr_grpo**(none) | 0.380 | 0.358 | 0.400 | 0.425 | 0.487 |
| **GSPO**(seq IS) | **0.290** ⚠️ | 0.305 | 0.276 | 0.355 | 0.500 |

- **dr_grpo ≈ GDPO 동률**: Δ+0.01·math+0.02·format+0.02 모두 **N=200 노이즈(±0.034) 내**. on-policy(0.487 vs 0.490)와 일치. → Stage-2 무차별, GDPO는 스케일차 큰 멀티리워드 타깃이라 **Stage-3(judge 1.0+format 0.2)엔 GDPO 채택 권고**.
- **⚠️ GSPO는 홀드아웃 열위(0.290)** — on-policy에선 최고(0.500)였으나 홀드아웃은 3종 최하위. **train-test 격차 −0.21**(dr_grpo −0.11의 2배) = **일반화 실패**. 시퀀스 IS(작은 ε)가 on-policy 분포엔 맞았으나 홀드아웃 일반화는 나쁨. → **"GSPO 미채택" 근거 강화**(단순 동률이 아니라 홀드아웃 명확 열위). **on-policy만 보면 오판했을 사례 — 홀드아웃 측정의 가치**.
- 부수 발견: dr_grpo/GDPO 둘 다 step600서 **~0.38–0.39**로 step800(0.38)과 동일 → **Stage-2 홀드아웃이 step600쯤 포화**.

<details><summary>⚠️ DAPO·baseline 홀드아웃 (오염·참고용) — 누수가 수치로 드러난 사례</summary>

DAPO(57527)·baseline(57249)은 홀드아웃 분리 前 **구 데이터**(deepvision103k_train, 홀드아웃 972건 **포함**)로 학습 → 홀드아웃 채점 = 학습셋으로 시험(**누수**). 완결성 위해 측정(`49_eval_contaminated.slurm`):

| 메소드 | 홀드아웃 Acc | (참고) on-policy | 데이터 |
|------|------|------|------|
| DAPO (step600) | 0.415 ⚠️ | 0.465 (미돌파) | 구(오염) |
| baseline (step1000) | 0.415 ⚠️ | 0.500 (plateau) | 구(오염) |

- **역설이 곧 누수 증거**: DAPO·baseline은 on-policy에서 **진 메소드**(plateau·미돌파)인데 홀드아웃은 clean 승자(dr_grpo 0.380)보다 **높다**(0.415). 더 약한 모델이 더 높게 나옴 = 홀드아웃 972건을 학습에서 암기(≈+0.03~0.04 인플레).
- → **0.415는 clean 3종과 같은 축 비교 불가**. 이 대비가 오히려 **층화 홀드아웃 분리의 정당성**을 입증(오염 모델이 부당히 높게 나옴).
</details>

<details><summary>이전 파일럿 수치(무효, 참고용) — DeepVision stride 100건, 오염·21%학습</summary>

| 지표 | base | 구 dr_grpo_merged | 변화 |
|------|------|------|------|
| 정확도 | 0.21 | 0.35 | +0.14 (누수·21%) |
| 형식 | 0.23 | 0.46 | — |
| 평균 길이 | 5618 | 4414 | — |

*평가셋이 학습 파일 stride 슬라이스라 진짜 홀드아웃 아니었음(누수) → 위 층화 홀드아웃 수치로 대체.*
</details>

---
