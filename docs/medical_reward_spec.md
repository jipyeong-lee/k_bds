# 3단계 의료 복합 보상 (clinical_judge) 구현 스펙

> 대상: `configs/medical_reward.py` + `scripts/30_medical_rl.slurm`
> 상태: **스펙(구현 전)**. ms-swift 4.1.3 컨테이너 소스 조사 기반.

## 1. 목표 & 범위
- **judge 는 3단계(개방형 의료 RL) 전용**. 2단계 RLVR(DeepVision)은 judge 안 씀 — 검증가능 정답을
  `accuracy`(math_verify, `\boxed{}` vs ground_truth) 로 규칙 채점(RLVR 의 정의). judge 불필요·부적합.
- 3단계 데이터 `MBZUAI/medix-rl-data`(자유형식 임상 QA, `solution`=참조답)는 **단일 정답이 없어**
  exact-match/규칙 채점 불가 → **LLM-as-judge** 로 임상 유효성 평가.
- 복합 보상: 형식(rule) + 임상 유효성(judge). 계획서 요구: 보상해킹·환각 억제, 다각도 평가.
- (참고) medix 일부 답은 측정값 등 준-검증가능 → 필요시 verifiable 부분집합에 `accuracy` 병용 가능(옵션).

## 2. 핵심 결정 — `AsyncORM` 로 외부 멀티모달 judge API 호출
judge 가 **(나중에) 대형·멀티모달 모델을 API 로 사용** 예정 → in-process 방식 부적합.
- ms-swift 네이티브 `GenRMPlugin` 은 judge 모델을 `TransformersEngine` 로 **in-process 적재** → 외부 API 와 맞지 않음(또 학습 GPU 점유). → **사용 안 함.**
- 대신 **`AsyncORM` 상속** (검증: GRPO 트레이너가 async ORM 을 `asyncio.gather` 로 병렬 호출, `reward_kwargs` 에 `images`/`solution` 등 데이터셋 컬럼 전달). judge API 를 async 호출 → 학습 GPU 미점유.
  - 등록 `orms['clinical_judge']`, 사용 `--reward_funcs format clinical_judge` + `--external_plugins configs/medical_reward.py`.
- 🚨 **제약(중요): 컴퓨트 노드 오프라인.** 상용 API(OpenAI/Gemini 등)는 학습 노드에서 **도달 불가**.
  → judge API 는 **클러스터 내부 self-host**(대형 멀티모달 VLM 을 별도 노드 GPU 에서 vLLM OpenAI 서버로 기동, 학습 노드가 내부 IP 로 호출) 여야 함. (또는 egress 예외 확보) — **선행 확인 필요(§9).**

## 3. 복합 보상 구성
| 소스 | 메커니즘 | 역할 | weight(초안) |
|------|----------|------|------|
| `format` | 내장 `Format` ORM (`--reward_funcs format`) | 출력 형식 게이트 | 0.1 |
| `clinical_judge` | `GenRMPlugin` 상속 (`--reward_model`+`--reward_model_plugin clinical_judge`) | 임상 유효성 0~1 | 1.0 |
| (옵션) `soft_overlong` | 내장 (`--reward_funcs soft_overlong`) | 장황함/길이 페널티 | 0.1 |

- `--reward_weights` 로 가중 합산. **길이 = reward_funcs 수 + reward_model 수** 와 정확히 일치해야 함.
  예: `--reward_funcs format` + `--reward_model <judge>` → 보상원 2개 → `--reward_weights 0.1 1.0`.
- 보상해킹 완화: 형식 위반 시 judge 점수를 0으로 게이팅(아래 4.3) + soft_overlong 로 장황 답변 억제.

## 4. clinical_judge 상세 (`ClinicalJudge(AsyncORM)`) — 멀티모달 API judge
파일: `configs/medical_reward.py` → 끝에 `orms['clinical_judge'] = ClinicalJudge`.

### 4.1 동작
- `async __call__(self, completions, **kwargs)`: kwargs 에서 `solution`(참조답), `images`(이미지), 질문(messages) 추출.
- 각 completion 마다 **멀티모달 judge API** 호출(이미지+질문+참조답+모델답변) → 0~1 점수 파싱 → asyncio.gather 병렬.
- 클라이언트: `AsyncOpenAI(base_url=$JUDGE_BASE_URL, api_key=$JUDGE_API_KEY)`, model=`$JUDGE_MODEL`.
  이미지는 OpenAI 멀티모달 포맷(`image_url` base64) 으로 전달.

### 4.2 루브릭(judge system 프롬프트) — 다차원, 보상붕괴 완화
1. **임상적 정확성** — 참조답(gold) 대비 핵심 소견/진단 일치
2. **사실성/환각 없음** — **이미지에 근거 없는 주장 페널티(멀티모달 judge 의 핵심 강점)**
3. **완전성** — 요구된 임상 요소 포함
4. **안전성** — 위험한 오답(놓친 위중 소견 등) 강한 페널티
→ 0~1 종합, 파싱 가능한 형식(예: `Reward: 0.85` 또는 JSON `{"score":..}`). 앵커 예시로 차등 점수 유도.

### 4.3 견고성
- 타임아웃/실패/파싱불가 → 0.0 반환(학습 중단 방지). 결정적 채점 위해 judge temperature=0.
- 형식 게이팅: 모델답이 `<answer>` 형식 위반이면 judge 호출 생략하고 0(비용 절감 + 형식 강제).

## 5. judge 서빙 & GPU/노드 예산 (오프라인 제약 반영)
- judge 는 **외부 API** → 학습 GPU 미점유(정책+참조+rollout 만 학습 노드 사용). 좋음.
- 그러나 **컴퓨트 노드 오프라인** → judge API 는 **클러스터 내부에서 self-host** 필요:
  - 별도 GPU 노드(예: 4gpu/8gpu)에서 **대형 멀티모달 VLM 을 vLLM OpenAI 서버**로 기동.
  - 학습 노드(stage3)가 그 노드의 **내부 IP:port** 로 호출(`JUDGE_BASE_URL`).
  - 비용: judge 서버 노드가 학습과 **동시 가동** → 추가 노드시간(예산 §노드시간 재산정 필요).
  - 선행: ① 노드 간 내부 네트워크 도달성 확인 ② judge 모델 사전 다운로드 ③ judge 서버 기동 스크립트.
- (상용 멀티모달 API 를 쓰려면 컴퓨트 노드 egress 예외가 있어야 함 — 기본은 불가.)

## 6. 통합 — `30_medical_rl.slurm` 인자
```bash
run_py swift rlhf --rlhf_type grpo \
  --model "$GEN_CKPT" --model_type "$MODEL_TYPE" \
  --dataset "$DATA_DIR/medix_rl_train.jsonl" \
  --reward_funcs format clinical_judge \
  --reward_weights 0.1 1.0 \
  --external_plugins "$PROJ_DIR/configs/medical_reward.py" \
  --use_vllm true --vllm_mode server ...
# 환경변수: JUDGE_BASE_URL(내부 vLLM judge), JUDGE_MODEL, JUDGE_API_KEY
```
- clinical_judge 는 `AsyncORM` 이므로 **`--reward_funcs` 에 직접 나열**(현 슬럼 스크립트와 일치). reward_model/plugin 인자 아님.

## 7. 출력 형식 컨벤션 (결정됨) — `<think>…</think><answer>\boxed{답}</answer>`
파이프라인 전체 통일. **검증 완료**:
- 내장 `Format` 정규식 `^<think>.*?</think>\s*<answer>.*?</answer>$` → 이 형식 매치(✓), 맨몸 `\boxed{}` 거부(✗).
- `MathAccuracy`(2단계)는 코드상 **`<answer>...</answer>` 추출 후** 내부 `\boxed{}` 를 math_verify 로 파싱 → 정합(✓).
- 따라서 2단계(`accuracy`+`format`)·3단계(`format`+judge) 가 동일 형식 공유.
- 시스템 프롬프트: "단계별 추론을 `<think></think>` 안에, 최종답을 `<answer>\boxed{…}</answer>` 안에" 로 통일.

### SFT 재구성 함의 (중요)
- 현재 SFT(`sft_train.jsonl`)는 `\boxed{C}`(맨몸) 형식 → **재빌드 필요**: assistant 타깃을
  `<think>{추론}</think><answer>\boxed{답}</answer>` 로.
- ⚠️ DeepVision/medix 에 **gold 추론(CoT)이 없음** → `<think>` 내용 문제. 선택:
  - (a) **CoT 데이터셋으로 SFT cold-start** (`UCSC-VLAA/VLAA-Thinking` 등 — 추론 트레이스 보유) → `<think>` 채움. **권장**.
  - (b) 최소/빈 `<think></think>` 로 형식만 SFT — 추론 학습 안 됨(권장 안 함).
  - (c) **SFT 생략**, GRPO 의 format+accuracy 보상으로 형식·추론 직접 유도(R1-zero 방식). 계획서엔 SFT 명시.
- → `build_sft.py` 에 `--answer-format think_boxed` 옵션 추가 + CoT 입력 지원(style=cot)로 처리 예정.

## 8. 검증 계획
- 단위: 샘플 답변군(정답/오답/빈답/장황답)에 judge 적용 → **점수 단조성**(정답>오답>빈답) 확인.
- 분포: 변환 데이터 100~200건 judge 점수 히스토그램 → 붕괴 여부.
- reward-hacking 프로브: 형식만 맞고 내용 틀린 답 → 낮은 점수 확인.
- 속도: step당 judge 지연 측정 → 병목 시 judge 축소/병렬/배치 조정.

## 9. 결정 현황
- ✅ 출력 형식: `<think>…</think><answer>\boxed{답}</answer>` (확정, §7).
- ✅ SFT: **생략하고 GRPO 직행**(R1-zero) — Qwen3.5-9B 가 추론·지시따르기 가능. 강한 시스템 프롬프트로 형식 유도,
       GRPO format 보상이 보강. 초기 GRPO 에서 format 보상이 낮으면 경량 format SFT 추가(가역적). 기존 sft_*.jsonl 은 폴백 보관.
- ✅ judge: **대형 멀티모달 모델 API**(AsyncORM, §2/§4). 3단계 전용.
- ⏳ 🚨 **judge API 도달성**(오프라인 컴퓨트 노드): 상용 API 불가 → **클러스터 내부 self-host vLLM** 으로 갈지,
       egress 예외가 있는지 **확인 필요**. 설계의 핵심 분기.
- ⏳ judge 모델 구체(어떤 대형 VLM), reward_weights, judge 서버 노드 할당/노드시간.
- 참고: 2단계(DeepVision RLVR)는 judge 불필요 → **지금 바로 진행 가능**.

## 10. 리스크
- judge 비용/속도(TransformersEngine) → 학습 throughput 저하.
- GPU OOM(정책+참조+rollout+judge 4모델 동시 적재).
- reward collapse / judge 편향·환각(임상 오판) → 강력 judge + 참조답 의존 + 다차원 루브릭으로 완화.
- judge 자기일관성 부족 → 온도 0 / 결정적 디코딩.
