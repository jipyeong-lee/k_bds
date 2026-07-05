# K-BDS 의료 멀티모달 교차추론 — 문제정의·실험결과·해결방안·목표

기준일 **2026-07-05**. base = `Qwen/Qwen3.5-9B`. Stage-2 init = `sft_rft_coldstart_merged`(SFT 콜드스타트 병합본, base 아님).

---

## 1) 문제 정의

**상위 목표**: base의 **의료 멀티모달 교차추론** 능력을 정량적으로 향상 — 특히 검증가능 정답이 없는 **개방형 의료 VQA**에서 "이미지를 실제로 보고 다기준 근거로 추론"하게 만든다.

**계획**: 4단계 파이프라인 ① 콜드스타트 SFT → ② 범용 RLVR/GRPO → ③ 의료 특화 RL(RaR) → ④ 평가.

| # | 하위 기술 문제 | 근거(관측) | 상태 |
|---|---|---|---|
| P1 | base가 장문 추론(3.5~4.6K토큰)으로 **답 잘림 → 0점** | RL 초기 clip↑, FormatThink 0.05 | ✅ SFT 해결 |
| P2 | Stage-2 범용 RLVR **Acc plateau** | `frac_reward_zero_std` 0.24→0.33, Acc 0.50 정체 | ✅ dr_grpo 돌파 |
| P3 | 의료 도메인은 **검증불가(개방형)** → RLVR 직접 적용 불가 | 단일정답 규칙채점 불가 | 🔄 RaR 배선완료·본실행 대기 |
| P4 | **NVLink 없음** → full-FT 통신 병목 | full 375~660s/step vs LoRA 128s | ✅ 전 단계 LoRA-DDP |
| P5 | 평가 **홀드아웃 누수** | stride 슬라이스가 학습셋과 겹침 | ✅ 층화 홀드아웃 분리 |

핵심 미해결 = **P3(의료 RL 본실행)** + **Stage-2 기법 최종 확정(GDPO 판정)**.

---

## 2) 현재까지 실험 결과 (정량)

### (a) Stage-1 SFT 콜드스타트 (job 57242, 727건 RFT 정제본, 4 epoch)
- FormatThink 보상 **0.05 → 0.27**(약 5배), clip↓ → 잘림 0점 문제 제거.
- 홀드아웃 기여: base 0.15 → **init(SFT) 0.22**(형식 0.12→0.32, 길이 5907→5188자).
- 구축 방식(2단계 진화) 상세: [부록 A](#부록-a-stage-1-sft-콜드스타트-구축-상세).

### (b) Stage-2 기법 A/B — 돌파판정 (step 501~600 구간평균)

| 기법 | init | Acc | zero_std | mean_len | 판정 |
|---|---|---|---|---|---|
| baseline GRPO (57249) | 콜드스타트 | 0.500 | 0.24→0.33↑ | 3267 | plateau |
| DAPO (57527) | 콜드스타트 | 0.465 | 0.00 | ~3600↑ | 미돌파 |
| **dr_grpo (57624)** ✅ | 콜드스타트 | **0.526** | 0.00 | 3259 | **돌파·승자** |
| GSPO (59004) | 콜드스타트 | 0.500 | — | — | 동률→미채택 |
| GDPO (59191, 진행중) | 콜드스타트 | AccMix 동급(~0.50) + **Format +0.04~0.06 우위** | 0.00 | — | 판정창 대기 |

- GSPO 판정: dr_grpo 0.500 vs GSPO 0.487 (±0.013 노이즈) → **동률, dr_grpo 유지**.
- GDPO 현재: **step 319/600(~53%)**, 잔여 ~1.2일. AccMix·총 reward dr_grpo 동급, FormatThink 우위 150스텝 지속, `zero_std=0`.

### (c) 🎯 중간 홀드아웃 벤치마크 — 가장 결정적 (RL 25%=step 800, 층화 N=100)

| 모델 | 전체 Acc | math | visual-logic | 형식 | 평균길이 |
|---|---|---|---|---|---|
| base | 0.15 | 0.19 | 0.11 | 0.12 | 5907자 |
| init (SFT 콜드스타트) | 0.22 | 0.23 | 0.21 | 0.32 | 5188자 |
| **trained (dr_grpo 25%)** | **0.38** | 0.34 | **0.42** | 0.45 | 4752자 |

→ **init→trained +73%**(RL 순효과), **base→trained +153%**(전 파이프라인), visual-logic +100%.

---

## 3) 해결방안 + 방안별 정량 평가방법

| 방안 | 내용 | 정량 평가방법 |
|---|---|---|
| **S1. Stage-2 기법 확정** | dr_grpo(승자) vs GDPO A/B 판정 | 판정창 501~600 구간평균 **Acc·reward** + 병합 후 **층화 홀드아웃**(math/vl 분리, N=100→972) |
| **S2. Stage-2 전량 학습** | 채택기법으로 1 epoch(≈3204 step) 완주 | 최종 ckpt **층화 홀드아웃 재측정**(정식 수치, 중간 0.38 대체) |
| **S3. Stage-3 의료 RL(RaR)** | medix-rl-data 51K, 정적 4차원 루브릭 보상 | **의료 VQA 홀드아웃 Acc** + **RaR clinical judge 점수**(0~1) base/init 대비 |
| **S4. 최종 벤치마크** | base vs 최종모델 | 층화 홀드아웃 + 의료 벤치, Acc/math/vl/형식/길이 전 지표표 |

평가 공통 원칙: **동일 조건**(시스템프롬프트·temperature 0·max_tokens) 채점, 진짜 홀드아웃(trainonly에서 제외한 972건, math 453 + vl 519), `accuracy_mix` 자동채점.

---

## 4) 방안별 예상결과(정량 목표) + 기한

| 방안 | 정량 목표 | 기한(2026-07 기준) |
|---|---|---|
| **S1. GDPO 판정** | GDPO Acc ≥ dr_grpo 0.487(동급) **AND** Format 우위 유지 → Stage-3 채택. 아니면 dr_grpo 유지 | **~07-07** (step 600 도달, +~1.2일) |
| **S2. Stage-2 완주** | 홀드아웃 Acc **0.38 → 0.42~0.45+**(math·vl 동반상승), 형식 ≥0.50 | 전량 ~10일 / 추세 정체 시 조기확정 |
| **S3. Stage-3 RaR** | clinical judge **base 대비 +유의미**, 의료 홀드아웃 Acc base 0.15 대비 **≥2배** | S2 확정 후 착수, ~1~2주 |
| **S4. 최종 보고** | base 대비 전 지표 향상표 + 사사표기 | 과제종료 **1주 내** 데이터 정리 · **1개월 내** 보고서·산출물 기탁 |

**임계경로**: `GDPO 판정(07-07) → 기법 확정 → Stage-2 완주/조기확정 → Stage-3 본실행 → 최종 벤치마크`.

> 목표값 주의: S2(0.42~0.45)·S3(≥2배)는 중간 벤치마크 추세 기반 **제안치**로, 확정 목표선은 조정 가능.

---

## 부록 A. Stage-1 SFT 콜드스타트 구축 상세

**왜 필요했나**: base(Qwen3.5-9B VL)가 어려운 시각/기하 문제에 본래 **매우 길게 추론** → 4096/6144 토큰 budget으로도 **~50% 잘림 → 무답 0점**. RL·budget 조정으로 못 잡음 → **간결 추론 습관을 SFT로 재주입**.

콜드스타트는 **2단계로 진화**했다.

### 1차 (plain) 콜드스타트 — 폐기
- 스크립트 `build_coldstart_sft.py`. 데이터 = `UCSC-VLAA/VLAA-Thinking`의 **clevr_math** 서브셋(~5.9K). answer가 이미 `<think>…</think><answer>X</answer>` 형식이라 거의 그대로 사용 → `sft_coldstart_train.jsonl` **2913건**.
- **문제**: clevr_math가 너무 쉬움(평균 760자) → 어려운 시각/기하 문제로 **일반화 실패**(여전히 잘림).

### 2차 (RFT) 콜드스타트 — 최종 채택 ✅
- 스크립트 `build_rft_coldstart.py` (STaR / ReST / RFT = **거절샘플링 자기증류**). 남의 데이터가 아니라 **모델 자신의 정답 롤아웃**을 모아 SFT → in-domain 간결 추론 재주입.
- **소스**: 자기 GRPO 롤아웃 로그 `grpo_general/v*/completions.jsonl`.
- **필터 3중**: ① `AccuracyMix==1.0`(정답) ② `<think></think><answer></answer>` **정상 마감**(비잘림) ③ `len ≤ 6000자`(≈2000토큰, 간결).
- **선별**: 질문당 가장 짧은 것부터 **최대 3개**(`PER_Q=3`). 이미지 경로는 DeepVision 원본과 질문텍스트로 join.
- **결과**: `sft_rft_coldstart_train.jsonl` **727건** + val 40건 (ms-swift `{messages, images}` 포맷).

### SFT 학습 (job 57242, 2026-06-16)
```
base Qwen3.5-9B + LoRA r16/a32 (all-linear)
727건 · 4 epochs · LR 1e-4 · max_len 4096 · bs1×grad_accum8 · 8GPU DDP
→ sft_rft_coldstart_lora/checkpoint-48
→ (merge, job 57245) sft_rft_coldstart_merged (18G)  ← Stage-2 init
```

### 학습곡선 (48 step = 12 step/epoch × 4, 총 ~5분)

| epoch/step | train loss | val loss | val token_acc |
|---|---|---|---|
| step 1 | 0.263 | — | — |
| epoch 1 (step 12) | ~0.23 | 0.2149 | 0.9237 |
| epoch 2 (step 24) | ~0.19 | 0.2136 | 0.9236 |
| epoch 3 (step 36) | ~0.20 | 0.2137 | 0.9238 |
| epoch 4 (step 48) | 0.197 | 0.2138 | 0.9242 |

- **형식은 ~1 epoch에 수렴**: val loss가 epoch 1부터 0.214로 평탄(이후 변화 ±0.001), token_acc 0.924 유지. 추가 epoch는 한계효용 작음(train 0.26→0.20으로 소폭 하락, val 정체 = 과적합 없음).
- 형식 주입(간결 `<think>/<answer>`)이 목적이라 val 조기수렴은 정상 — 새 지식 학습이 아니라 **출력 스타일 정렬**.

### 727건 샘플·길이 분포
- 예시(질문): *"Jewel2 … 특수 원소가 몇 개인가"* (시각논리 퍼즐) → assistant `<think>보드 스캔·계산…</think><answer>N</answer>`.
- **assistant 완성문 길이(문자)**: 중앙값 **2837** · 평균 2861 · 최소 36 · 최대 5979 → `≤6000자` 필터 정상 작동, base 폭주(3.5~4.6K토큰) 대비 **절반 수준의 간결 추론**.

### 검증
- probe 추론(`probe_coldstart_think_true/false.jsonl`, `probe_rft_think_true.jsonl`)으로 형식 준수 확인.
- 후속 GRPO **FormatThink 0.05→0.27**(≈5배)·clip↓ = 잘림 0점 해소.
- 중간 홀드아웃 **base 0.15 → init 0.22**(형식 0.12→0.32, 길이 5907→5188자)로 SFT 기여 격리 확인.

**한 줄**: 쉬운 외부데이터(clevr_math) 형식 콜드스타트가 일반화에 실패 → **어려운 in-domain(DeepVision)에서 모델 자신의 "정답+마감+간결" 완성문 727개를 거절샘플링으로 뽑아 4-epoch LoRA SFT**로 간결 추론 습관 주입.
