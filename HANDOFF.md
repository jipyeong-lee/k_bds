# HANDOFF — K-BDS 의료 멀티모달 교차추론 학습 파이프라인

> 갱신: 2026-06-29 / 다음 작업자(또는 다음 세션)를 위한 인수인계.
> 상세 배경은 `README.md`(현황·의사결정·섹션별)와 `docs/worklog_*.md`(일별) 참조.

---

## 1. 현재 위치 (TL;DR)

- **Stage-2(범용 RLVR/GRPO) 종결.** GRPO 파생기법 A/B 결과 **dr_grpo 가 승자**.
  - GRPO baseline(57249): **Acc plateau**(zero_std 0.24→0.33).
  - DAPO(57527): 안정성 압도하나 **돌파 미확인**(step501~600 Acc 0.465<baseline 0.500, 길이 재폭주).
  - **dr_grpo(57624): ✅ 돌파 확인**(step501~600 Acc **0.526**>0.500, 길이 억제 동반) → **승자 산출물 `checkpoint-600`**.
- **Stage-3(의료 RL) 진행 중 — RaR 루브릭 보상 + judge 검증까지 완료.**
- **현재 실행 중인 잡 없음.** (watcher 도 모두 종료) 다음은 judge 도달성 테스트 → 분포 프로브 → Stage-3 본실행 배선.

---

## 2. Stage-2 최종 결과 (완결)

| | baseline | DAPO | **dr_grpo (승자)** |
|---|---|---|---|
| Job | 57249 (step1000 완주) | 57527 (step623 scancel) | 57624 (step689 TIMEOUT) |
| 출력 | `grpo_general/v11-.../checkpoint-1000` | `grpo_general_adv_dapo/v1-.../checkpoint-600` | **`grpo_general_adv_dr_grpo/v0-20260625-081142/checkpoint-600`** |
| 레시피 차이 | 표준 GRPO | `loss_type=dapo`+clip-higher(0.28) | `loss_type=dr_grpo`+`scale_rewards=none` |
| 공통 코어 | — | `dynamic_sample`+`overlong_filter` | `dynamic_sample`+`overlong_filter` |
| step501~600 Acc | 0.500 | 0.465 ❌ | **0.526 ✅** |

- 3기법 메커니즘·실증 비교는 README "Stage-2 A/B" 절(표·plot) 참조. A/B 자동추적 인프라: `scripts/grpo_watch.sh`+`grpo_ab_update.py`+`plot_grpo_compare.py`(레시피 일반화, watcher가 100-step마다 README·plot 갱신·push). **현재 watcher 비가동**(Stage-2 종료).
- **Stage-3 init = dr_grpo `checkpoint-600`** (이 LoRA 어댑터를 base 병합 후 init 으로 사용 예정).

---

## 3. Stage-3 — RaR 루브릭 보상 (설계 확정·judge 검증 완료)

개방형 의료 VQA(medix)는 단일정답 규칙검증 불가 → **Rubric-as-a-Reward**(arXiv:2507.17746). 상세 `docs/medical_reward_spec.md`.

- **데이터**: `work/data/medix_rl_train.jsonl`(51K, 단답 의료 VQA. 정답 중앙값 46자, 예 "28×27mm"·"X-ray").
  - RaR-Medicine-20k 는 텍스트전용·장문이라 **비채택**(스키마만 차용).
- **루브릭(정적 4차원, 가중=RaR 정수)**: 정답정확성(5) / 시각근거(3,`<think>`·교차추론) / 정밀도·단위(3,측정형 자동분기) / 환각Pitfall(4).
  단답이라 핵심사실=참조답 1개 → **참조답을 Essential 기준에 템플릿 주입**(LLM 루브릭 생성 불필요). explicit 집계 `r=Σwⱼcⱼ/Σwⱼ`.
- **구현** `configs/medical_reward.py` `ClinicalJudgeReward(AsyncORM)`: 형식게이트→멀티모달 judge API→JSON 0/1 파싱→집계. 타임아웃/파싱실패→0.0.
  - 등록 `orms['clinical_judge']`, 사용 `--reward_funcs format_think clinical_judge --external_plugins configs/medical_reward.py`.
  - env 주입: `JUDGE_BASE_URL`(예 `http://<judge노드IP>:8100/v1`) / `JUDGE_MODEL=qwen36-judge` / `JUDGE_API_KEY`.
  - 유닛테스트 `scripts/test_medical_reward.py`(swift 스텁+mock judge) **24/24** — 컨테이너 없이 `python3`로 실행.

### judge 모델 = `Qwen/Qwen3.6-27B-FP8` (검증 완료)
- 멀티모달, Apache-2.0, ~31GB. **config 실측 `model_type=qwen3_5` = base 와 동일 arch** → 컨테이너 vLLM 0.19.1 그대로 서빙.
  `work/hf_cache` 에 다운로드 완료.
- 서버: `scripts/judge_server.sh` (40GB 보수설정 maxlen8K·enforce-eager·util0.92·TP). 스모크: `scripts/31_judge_smoke.slurm`+`judge_smoke_client.py`.
- **스모크 결과(job 58296, 1gpu)**: FP8 30GB **단일 40GB 적합**(36.6GB), 멀티모달 채점·JSON 파싱 OK, **정답1.0>오답0.0 단조성 PASS**.

---

## 4. 🚨 핵심 제약 / 함정 (Lessons learned)

- **NVLink 없음** → ZeRO-3/full-FT 멀티GPU 5배 느림(불채택). **전 단계 LoRA-DDP**.
- **로그인노드는 vLLM 불가**: 드라이버 **470(CUDA11.4)** → `cuTensorMapEncodeTiled` 없어 `vllm._C` 로드 실패.
  GPU(A100 40GB) 있어도 못 씀. **모든 GPU 추론/학습은 컴퓨트노드(드라이버550)**. ← judge 도 컴퓨트노드.
- **컴퓨트노드 외부망 차단(오프라인)**: 외부 API judge 불가 → 내부 self-host. 단 **컴퓨트노드 간 내부망은 열림**(judge↔학습 내부 IP 호출) — 도달성은 테스트로 최종확인 필요.
- **Qwen3 judge 는 추론모델** → 기본 응답 content 비어 파싱 0 됨. `enable_thinking=False`(chat_template_kwargs)로 끄고 JSON만 받음(반영됨).
- **vLLM 0.19.1 `--limit-mm-per-prompt` 는 JSON 문법**(`'{"image":1}'`, `image=1` 아님).
- 단일 step 지표 노이즈 큼 → 50/100-step **구간평균**으로 판단.
- 출력 형식 통일: `<think>…</think><answer>…</answer>` (Stage-2·3 공통, format_think 게이팅).

---

## 5. 다음 할 일 (순서)

1. [ ] **학습↔judge 내부망 도달성 테스트** — judge 잡(1gpu/2gpu) 먼저 띄우고, 별도 잡(다른 컴퓨트노드)에서 그 노드 IP:port 접속 확인.
2. [ ] **분포 프로브**(spec §8) — medix 100~200건에 judge 적용 → 점수 분포·단조성·**c2(시각근거) 캘리브레이션** 점검(스모크에선 c2 다소 관대).
3. [ ] **`scripts/30_medical_rl.slurm` 배선** — init=dr_grpo `checkpoint-600`(병합본), `--reward_funcs format_think clinical_judge`,
   `--reward_weights`(초안 0.2 1.0), judge 노드 `JUDGE_BASE_URL` 자동 주입. judge 잡 + 학습 잡 동시 가동(노드시간 추가 — 예산 재산정).
4. [ ] **Stage-3 본실행** — medix(+DeepVision 일부 혼합으로 망각 방지) LoRA GRPO.
5. [ ] `scripts/40_eval.slurm` EVAL_DATASETS 를 실제 의료 멀티모달 벤치마크로 교체.

---

## 6. 실행 / 환경 메모

- **클러스터**: KISTI K-BDS Slurm. 파티션: `1gpu`/`2gpu`(A100 **40GB**), `4gpu`/`8gpu`(A100 **80GB**). 학습=8gpu, judge=1gpu/2gpu.
- **컨테이너**: `work/images/ms-swift-413-sandbox`(swift4.1.3/torch2.10/vllm0.19.1). glibc2.17이라 conda vLLM 불가.
- **공통설정**: `scripts/00_common.sh`. 단일GPU 디버그는 `export NPROC_PER_NODE=1` 필수.
- **judge 서버 기동**(컴퓨트노드): `JUDGE_PORT=8100 JUDGE_TP=1 bash scripts/judge_server.sh` (1gpu) / `JUDGE_TP=2`(2gpu).
- **푸시**: `git push origin HEAD:master` (로컬 main → 원격 master).
- ⚠️ **보안**: `~/model_download.py` 에 HF 토큰 평문 노출 — 재발급·env 분리 권장.

---

## 7. 자산 위치

- **Stage-3 init**: `work/checkpoints/grpo_general_adv_dr_grpo/v0-20260625-081142/checkpoint-600` (dr_grpo 승자 LoRA).
- **judge 모델**: `work/hf_cache/.../models--Qwen--Qwen3.6-27B-FP8` (30GB, 다운로드 완료).
- **데이터**: `work/data/{medix_rl_train.jsonl, deepvision103k_train.jsonl}`.
- **스펙/일지**: `docs/medical_reward_spec.md`, `docs/worklog_2026-06-*.md`(15~29).
- **메모리**: `~/.claude/.../memory/` (클러스터 환경·과제목표·glibc/컨테이너 제약).
