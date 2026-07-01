# HANDOFF — K-BDS 의료 멀티모달 교차추론 학습 파이프라인

> 갱신: 2026-06-30 / 다음 작업자(또는 다음 세션)를 위한 인수인계.
> 상세 배경은 `README.md`(현황·의사결정·섹션별)와 `docs/worklog_*.md`(일별) 참조.

---

## 1. 현재 위치 (TL;DR)

- **Stage-2(범용 RLVR/GRPO) — dr_grpo 로 1 epoch 재학습 진행 중.**
  - GRPO 파생기법 A/B 결과 **dr_grpo 가 승자**(baseline·DAPO 는 plateau 미돌파). → §2.
  - ⚠️ **직전까지의 dr_grpo 런은 두 문제로 재학습 중**: ① 데이터의 **21%(step~650)만** 학습(중간 중단), ② 평가가 학습 파일 stride 슬라이스라 **진짜 홀드아웃이 아님**(누수).
  - **조치**: 정답유형 **층화 홀드아웃 분리** + **init 부터 trainonly 로 fresh 1 epoch 재학습**(누수 0). → §2.
  - **현재 실행 중**: job **58892**(dr_grpo, trainonly, MAX_STEPS=3204) + afterany 체인 5잡(총 6잡, QOS 상한).
- **Stage-3(의료 RL, RaR) — 설계·검증 완료, Stage-2 완주까지 대기.** 루브릭 보상·judge 검증·배선 끝. 정적 vs 인스턴스 루브릭 실증 비교로 **정적 채택 확정**. → §3.

---

## 2. Stage-2 — 범용 RLVR (GRPO 파생기법 A/B → dr_grpo 재학습)

### 2.1 기법 A/B 결론 (완결)

| | baseline | DAPO | **dr_grpo (승자)** |
|---|---|---|---|
| 레시피 차이 | 표준 GRPO | `loss_type=dapo`+clip-higher(0.28) | `loss_type=dr_grpo`+`scale_rewards=none` |
| 공통 코어 | — | `dynamic_sample`+`overlong_filter` | `dynamic_sample`+`overlong_filter` |
| step501~600 Acc(학습) | 0.500 | 0.465 ❌ | **0.526 ✅** |
| 진단 | plateau(zero_std 0.24→0.33) | 돌파 미확인·길이 재폭주 | **돌파**(길이 억제 동반) |

- 메커니즘: dr_grpo 는 loss_type(길이정규화 편향)·scale_rewards=none(난이도 편향) 두 편향 제거로 plateau 직격.
- 스크립트: `scripts/21_rlvr_grpo_adv.slurm`(RECIPE=dapo|gspo|dr_grpo, RESUME/MAX_STEPS 지원).

### 2.2 🚨 홀드아웃 정비 + 1 epoch 재학습 (현재 작업)

- **문제**: 기존 평가(`eval_compare.py`)가 학습 파일 `deepvision103k_train.jsonl` 의 stride(`i%137==11`) 슬라이스라 **분리 보장 없음**. 1 epoch 가면 평가샘플 전부 학습 포함 → 암기 측정.
- **분리** (`scripts/make_holdout.py`, 재현 가능): DeepVision 엔 카테고리 라벨 없음 → **정답유형을 math/visual-logic 프록시**로.
  - `math`(수치+수식) 45,284 / `vl`(객관식 MC) 51,896 / `other`(모호) 6,323.
  - 각 층 **1%** 추출 → **`deepvision_holdout.jsonl` 972**(math 453 + vl 519), 나머지 `deepvision103k_trainonly.jsonl` **102,531**. 합계 103,503 검증.
- **재학습**: init=`sft_rft_coldstart_merged` 에서 **fresh**(체크포인트 resume 아님 = 홀드아웃 미관측 보장), `DATASET=trainonly`, `MAX_STEPS=3204`(=1 epoch, 102531/32).
  - 출력 `work/checkpoints/grpo_general_adv_dr_grpo_he/`. 6잡 afterany 체인(58892~58982).
- **평가**: `eval_compare.py` → holdout 전용(stride 제거) + **math/vl 층별 정확도 분리 보고**(`_stratum` 필드).
- **속도/병목 진단**: 실측 ~365s/step(step_time 185 + 생성·동기화 ~180). **병리적 정체 아님** — 긴 생성(max_completion 6144, num_gen 4) + 40GB colocate(vllm_util 0.4) + NVLink 부재의 구조적 비용. 6잡 체인으로 완주 여유 확보.

### 2.3 ⚠️ 무효화된 이전 수치

- 이전 "base→학습모델 정확도 0.21→0.35(+67%)"(구 checkpoint-600, 구 stride 평가)는 **in-distribution 오염 + 21% 학습** 이라 **무효**. 1 epoch 완주 후 **층화 홀드아웃에서 재측정**해 교체.
- 구 산출물 `work/checkpoints/dr_grpo_merged`(구 checkpoint-600 병합)도 재학습 최종본으로 교체 예정.

---

## 3. Stage-3 — RaR 루브릭 보상 (설계 확정·judge 검증 완료, 대기)

개방형 의료 VQA(medix)는 단일정답 규칙검증 불가 → **Rubric-as-a-Reward**(arXiv:2507.17746). 상세 `docs/medical_reward_spec.md`.

- **데이터**: `work/data/medix_rl_train.jsonl`(51,335, 단답 의료 VQA. 예 "28×27mm"·"X-ray"). RaR-Medicine-20k 는 텍스트전용·장문이라 스키마만 차용.
- **루브릭(정적 통일 4차원, 가중=RaR 정수)**: 정답정확성(5,참조답 주입) / 시각근거(3,`<think>`) / 정밀도·단위(3,측정형 자동분기) / 환각Pitfall(4). explicit 집계 `r=Σwⱼcⱼ/Σwⱼ`.
- **정적 vs 인스턴스(논문식) 실증 비교** (`scripts/34_rubric_compare.slurm`, medix 40샘플): 항목수 비슷(3.0 vs 2.7)이나
  - 오답 기각 **정적 0.000 vs 인스턴스 0.118**, 환각변별(good−halluc) **정적 +0.338 vs 인스턴스 +0.021**.
  - 인스턴스는 이미지 없이 참조답만으로 생성돼 **시각근거 항목을 못 만듦** → **정적 통일 루브릭 채택 확정**.
- **구현** `configs/medical_reward.py` `ClinicalJudgeReward(AsyncORM)`: 형식게이트→멀티모달 judge API→JSON 0/1 파싱→집계. 유닛테스트 24/24.
  - 사용 `--reward_funcs format_think clinical_judge --external_plugins configs/accuracy.py configs/medical_reward.py --reward_weights 0.2 1.0`.
  - env: `JUDGE_BASE_URL`/`JUDGE_MODEL=qwen36-judge`/`JUDGE_API_KEY`.

### judge 모델 = `Qwen/Qwen3.6-27B-FP8` (검증 완료)
- 멀티모달, **`model_type=qwen3_5` = base 와 동일 arch** → 컨테이너 vLLM 0.19.1 그대로 서빙. `work/hf_cache` 다운로드 완료(~30GB).
- 서버 `scripts/judge_server.sh`(40GB 보수설정). 스모크(58296): FP8 36.6GB 단일40GB 적합, 멀티모달 채점·JSON OK, 정답1.0>오답0.0 단조성 PASS.
- 분포 프로브(33): good0.96/wrong0.00/halluc0.64, 단조성99%, c2변별 good0.94 vs halluc0.00(c2 완화 반영).
- 내부망 도달성(32): hostname·IP 200 통과.

---

## 4. 🚨 핵심 제약 / 함정 (Lessons learned)

- **NVLink 없음** → ZeRO-3/full-FT 5배 느림(불채택). **전 단계 LoRA-DDP**.
- **로그인노드 vLLM 불가**: 드라이버 470(CUDA11.4) → `cuTensorMapEncodeTiled` 없어 로드 실패. **모든 GPU 추론/학습은 컴퓨트노드(550)**. judge 도 컴퓨트노드.
- **컴퓨트노드 외부망 차단**(오프라인): 외부 API judge 불가 → 내부 self-host. **노드 간 내부망은 열림**.
- **Qwen3 judge 는 추론모델** → `enable_thinking=False`(chat_template_kwargs)로 끄고 JSON만 받음.
- **vLLM 0.19.1 `--limit-mm-per-prompt` 는 JSON 문법**(`'{"image":1}'`).
- **base 평가는 로컬 스냅샷 경로**로(HF id 주면 컨테이너가 modelscope 접근→read-only 실패). `VLLM_USE_MODELSCOPE=False`.
- **swift GRPO 는 reward 플러그인에 `images` 를 str 경로가 아니라 dict `{'bytes':..,'path':..}` 로 넘김**(스모크 실측). 커스텀 보상은 이 형식을 처리해야 함(안 하면 이미지 누락·시각근거 채점 blind). `medical_reward.py:_image_to_data_url` 가 str/dict/PIL 모두 처리. reward kwargs 실측 키: `['finish_reason','images','is_truncated','messages','prompt_id','request_id','response_token_ids','rollout_logprobs','solution','trainer_state']`.
- **Stage-3 배선 스모크**(`35_stage3_smoke.slurm`): idle 8gpu 노드에서 GPU0=judge/GPU1=소형 트레이너(max_steps 2). `JUDGE_DEBUG=1` 로 images kwarg 도달 검증. **PASS 확인**: data_url_ok=True, ClinicalJudgeReward 0.58→1.0, reward=0.2·Format+1.0·Clinical 정확 통합.
- **평가는 반드시 학습 미관측 홀드아웃에서** — stride 슬라이스는 학습 파일과 겹침(누수). §2.2.
- 단일 step 지표 노이즈 큼 → 구간평균으로 판단. 출력 형식 `<think>…</think><answer>…</answer>` 공통.

---

## 5. 다음 할 일 (순서)

1. [진행중] **Stage-2 dr_grpo 1 epoch 완주** — 58892~58982 체인(trainonly, MAX_STEPS=3204). 모니터: `squeue -u $USER`, `ls work/checkpoints/grpo_general_adv_dr_grpo_he/*/checkpoint-*`.
2. [ ] **최종 checkpoint 재병합** → `dr_grpo_merged` 갱신 (`merge_drgrpo.slurm ADAPTER=<최종ckpt>`).
3. [ ] **층화 홀드아웃 벤치마크** (`40_eval_compare.slurm`): base vs 학습본, **math/visual-logic 분리** 정확도 → README 정식 수치로 교체.
4. [ ] **Stage-3 init 교체**(→ 재병합본) 후 **Stage-3 본실행** `bash scripts/launch_stage3.sh`(judge 1gpu + 학습 8gpu 동시, 끝나면 judge scancel).
5. [ ] `40_eval` 의료 멀티모달 벤치마크 교체.

---

## 6. 실행 / 환경 메모

- **클러스터**: KISTI K-BDS Slurm. 파티션 `1gpu`/`2gpu`(A100 40GB)·`4gpu`/`8gpu`(80GB). 학습=8gpu, judge/eval=1gpu. **QOS 사용자당 제출 상한 6잡**.
- **컨테이너**: `work/images/ms-swift-413-sandbox`(swift4.1.3/torch2.10/vllm0.19.1). glibc2.17이라 conda vLLM 불가.
- **공통설정**: `scripts/00_common.sh`. 단일GPU 디버그 `export NPROC_PER_NODE=1`.
- **홀드아웃 재생성**: `python3 scripts/make_holdout.py`.
- **1 epoch 재학습 체인 제출**(참고): `--export=ALL,RECIPE=dr_grpo,[RESUME=1,]DATASET_FILE=<trainonly>,OUTPUT_DIR=<_he>,MAX_STEPS=3204` 로 첫잡 fresh + afterany 체인.
- **Stage-3 실행**: `bash scripts/launch_stage3.sh`. **푸시**: `git push origin HEAD:master`.
- ⚠️ **보안**: `~/model_download.py` HF 토큰 평문 노출 — 재발급·env 분리 권장.

---

## 7. 자산 위치

- **Stage-2 재학습 출력**: `work/checkpoints/grpo_general_adv_dr_grpo_he/` (진행중, save_steps=50).
- **Stage-3 init(예정)**: 위 재학습 최종 checkpoint 병합본(= `dr_grpo_merged` 갱신).
- **judge 모델**: `work/hf_cache/.../models--Qwen--Qwen3.6-27B-FP8` (~30GB).
- **데이터**: `work/data/{deepvision103k_trainonly.jsonl, deepvision_holdout.jsonl, medix_rl_train.jsonl}`. 분리기 `scripts/make_holdout.py`.
- **스펙/일지**: `docs/medical_reward_spec.md`, `docs/worklog_2026-06-*.md`.
- **메모리**: `~/.claude/.../memory/` (클러스터 환경·과제목표·glibc/컨테이너 제약).
