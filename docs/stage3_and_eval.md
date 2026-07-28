# Stage-3 · 의료 RL (RaR) & HealthBench 평가

> 이 문서는 [`README.md`](../README.md) 에서 분리된 상세 기록입니다. 요약·현황은 README, 상세는 여기.

## Stage-3 · 의료 RL (RaR 루브릭 보상)

개방형 의료 VQA(medix)는 단일정답 규칙검증 불가 → **Rubric-as-a-Reward**([arXiv:2507.17746](https://arxiv.org/abs/2507.17746)): judge가 가중 다기준 체크리스트를 항목별 0/1 채점, explicit 집계 `r = Σwⱼcⱼ / Σwⱼ ∈ [0,1]`(부분점수 dense). 상세 `docs/medical_reward_spec.md`.

### 루브릭 구성 (`configs/medical_reward.py` `build_rubric`)

**의도**: medix는 **단답 + 멀티모달**이라 "정답 하나 맞히면 끝"이 아니라 **"이미지를 실제로 보고 근거를 대며 맞혔는가"**를 나눠 채점 → 부분점수(dense) 보상. judge LLM이 아래 항목을 0/1 판정.

| 키 | 항목 | 유형(가중) | 판정 기준(요지) |
|---|---|---|---|
| **c1** | Answer correctness | Essential **5** | `<answer>`가 **참조답과 의미 일치**(동의어·단위환산·paraphrase 허용). 참조답을 기준 문장에 **직접 주입** |
| **c2** | Visual grounding | Important **3** | `<think>`가 **이미지와 모순 없는 관찰**을 언급·답 뒷받침. 없는 구조·풍경을 지어내면 실패 |
| **c3** | Numeric precision & unit | Important **3** | 수치+단위가 참조답의 **±15% 이내** (측정형 문제만 자동 추가) |
| **c4** | No hallucination / overclaim | Pitfall **4** | 이미지에 없는 소견·질문 범위 밖 주장 안 함 (긍정형) |

**집계**: `r = Σwⱼ·cⱼ / Σwⱼ ∈ [0,1]`. 실제 비중 — **비측정형**(분모 12): c1 **42%**·c2 25%·c4 33% / **측정형**(분모 15): c1 33%·c2 20%·c3 20%·c4 27%. → 정답성(c1) 최대, 환각억제(c4) 차순.

**설계 결정**:
- **정적 통일 루브릭**(문제마다 생성 X) — 인스턴스식(논문 방식)과 실증 비교(medix 40)에서 정적이 오답기각(0.000 vs 0.118)·**환각변별(+0.338 vs +0.021)** 우세. 인스턴스는 이미지 없이 텍스트로 생성돼 **시각근거 항목을 못 만듦** → **정적 채택**.
- **참조답 템플릿 주입**: 단답이라 핵심사실=참조답 1개 → c1 기준에 참조답을 박아넣어 **오프라인 LLM 루브릭 생성 불필요**(비용 0·결정적).
- **측정형 자동 분기**(`is_measurement`): 정답에 의료 단위(mm/cm/HU/° 등) 있으면 c3 자동 편입.
- **형식 게이트**(`format_ok`): `<think>`에 실질 추론(≥15자)+`<answer>` 있어야 judge 호출, 위반 시 **judge 생략하고 0.0**(비용↓·reward-hacking 방지).
- **c2 완화 이력**: 초기 "실제 소견 인용"은 너무 엄격해 항상 0 → **"이미지와 모순 없는 관찰"**로 완화(분포 프로브 good0.94 vs halluc0.00).
- **데이터**: medix = 단답 VQA(정답 중앙값 46자, 예 "28×27mm"). RaR-Medicine-20k(텍스트전용·장문)는 **스키마만 차용**.

상세 스펙은 `docs/medical_reward_spec.md` §4.2.

### 구현 & judge
- `configs/medical_reward.py` `ClinicalJudgeReward(AsyncORM)`: 형식게이트→멀티모달 judge(env `JUDGE_BASE_URL/MODEL/API_KEY`)→JSON 0/1 파싱→집계. 타임아웃·파싱실패→0.0. 사용 `--reward_funcs format_think clinical_judge --external_plugins configs/accuracy.py configs/medical_reward.py --reward_weights 0.2 1.0`.
- **judge = `Qwen/Qwen3.6-27B-FP8`**(멀티모달, 같은 `qwen3_5` arch → 컨테이너 vLLM 그대로 서빙). `scripts/judge_server.sh`.
  - 🚨 **로그인노드 불가**(드라이버 470 → vLLM 로드 실패). judge·학습 모두 **컴퓨트노드**(내부망 통신, 외부 egress 불필요).

### 검증 (전부 완료)
- ✅ 유닛테스트 **29/29** · judge 스모크(FP8 단일40GB 적합·정답1.0>오답0.0) · 내부망 도달성 · 분포 프로브(good0.96/wrong0.00, c2변별Δ0.94)
- ✅ **GRPO 배선 end-to-end 스모크**(`35_stage3_smoke.slurm`): 플러그인 로드·`clinical_judge` 해석·kwargs 도달·보상 통합 실증.
  - 🐛 **실버그 발견·수정**: swift가 `images`를 str 경로 아닌 **dict `{bytes,path}`**로 넘김 → `_image_to_data_url` str/dict/PIL 지원으로 수정(안 하면 시각근거 c2 blind). 재검증 PASS(data_url_ok=True, ClinicalJudge 0.58→1.0, reward=0.2·Format+1.0·Clinical 정확 통합).
- **RL 알고리즘**: RaR는 보상 설계일 뿐 → 최적화는 GRPO 계열. **Stage-3는 `scale_rewards=gdpo`(GDPO) 권고** — judge 1.0 + format 0.2의 **스케일차 큰 멀티리워드**가 GDPO 타깃(Stage-2 A/B서 downside 없음 확인, [판정](stage2_experiments.md#stage-2-3종-홀드아웃-판정-step600-n200)). dr_grpo 코어(`loss_type=dr_grpo`·dynamic_sample) 유지.

### 남은 것 (다음 임계경로)
⏳ **Stage-2 step600 체크포인트(dr_grpo 또는 GDPO) 병합 → init 교체 → `bash scripts/launch_stage3.sh`**(judge ready → 학습 자동 제출). GDPO 레시피 적용 권고. 망각 방지용 DeepVision 혼합은 옵션.

---

## 타겟 벤치마크 (HealthBench) — 의료 성능 측정

Stage-2 홀드아웃(DeepVision)은 **검증가능 정답** 기준의 내부 지표이고, **최종 의료 능력**은 외부 의료 벤치 **HealthBench**([OpenAI, 2025-05](https://openai.com/index/healthbench/), 262명 의사·5,000 대화·루브릭 채점)로 측정한다. 파이프라인(SFT→RLVR→RaR) 각 단계 모델을 **동일 하니스**로 재면서 base 대비 개선을 추적한다.

**측정 세팅**(`scripts/45_healthbench_smoke.slurm` + `run_healthbench.py`, evalscope `health_bench`):
- **Hard 서브셋 전량 1,000건**(가장 변별력 높은 부분집합), max_tokens 8192.
- **채점자 = 자체 Qwen3.6-27B-FP8**(오프라인 클러스터라 GPT-4.1 불가, thinking off). 데이터는 로컬 캐시(오프라인).
- `HB_TARGET_MODEL` env 로 평가대상 교체 → base/콜드스타트/Stage-2·3 모델 동일 조건 비교.

### 단계별 추적 — HealthBench Hard (n=1000)

| 모델 | **종합** | comm | instr | acc | complete | context | judge실패 | 평균길이 |
|------|------|------|------|------|------|------|------|------|
| **base** (Qwen3.5-9B) | **0.229** | 0.580 | 0.489 | 0.351 | 0.235 | 0.098 | 9.9% | 10,689자 |
| **① 콜드스타트 SFT (v2)** | 0.224 | 0.551 | **0.533** | 0.342 | 0.218 | 0.091 | 5.0% | **8,850자** |
| **① 콜드스타트 SFT (v3, 현행)** | *미측정* | | | | | | | |
| ② Stage-2 (dr_grpo/GDPO) | *예정* | | | | | | | |
| ③ Stage-3 (RaR) | *예정* | | | | | | | |

*평균길이 = 답변 문자수(HealthBench Hard 1000건). base는 최대 44,521자·24K자↑ 7.8%로 폭주 → 콜드스타트가 ~17% 단축(간결화).*

<details><summary>base 주제(theme)별 점수</summary>

| 주제 | 점수 (n) | | 주제 | 점수 (n) |
|------|------|---|------|------|
| emergency_referrals | 0.249 (66) | | health_data_tasks | 0.237 (115) |
| context_seeking | 0.248 (179) | | global_health | 0.229 (280) |
| hedging | 0.246 (167) | | complex_responses | **0.139** (82) |
| communication | 0.217 (111) | | | |
</details>

- **base 프로파일**: 말은 잘함(comm 0.58)·지시 따름(0.49)이나 **맥락인지(0.10)·완결성(0.24)·복잡시나리오(0.14)가 약함** — "그럴듯하나 상황을 온전히 파악·완결 못 함" = 의료 특화 학습 전 base의 전형. (Stage-3 RaR 루브릭이 겨냥하는 바로 그 결함)
- **콜드스타트 = 종합 동률(−0.005, 노이즈 내)**: SFT는 **형식·간결성 정렬**이지 의료 지식 주입이 아니므로 텍스트 의료 QA는 불변이 정상. 단 **instr +0.044**(지시 따르기 개선)·출력 간결화(평균 8.9K자, judge실패 9.9%→5.0%)로 프로파일만 이동. 콜드스타트의 본래 목적은 **DeepVision 시각추론 RL 인에이블**(→ 의료 성능은 Stage-2/3 RL의 몫).
- ⚠️ **해석 주의**: ① 채점자 ≠ GPT-4.1 → **공식 리더보드와 직접 비교 불가**(내부 상대비교 전용) ② judge 파싱실패 보수적 처리 → **완만한 하한** ③ base 장황(최대 44K자)으로 일부 잘림.

> **추적 계획**: Stage-2(dr_grpo/GDPO step600) → Stage-3(RaR) 모델을 동일 하니스로 측정해 위 표 ②③ 행 채움 → base 0.229 대비 단계별 개선 정량화.

---
