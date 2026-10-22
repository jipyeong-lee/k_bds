# 작업 일지 모음 — 2026-06-15 ~ 2026-07-01

> 일별 worklog 10개를 원문 그대로 한 파일로 합쳤다. 각 날짜 절의 본문은 원래 파일과 같다(제목 줄만 절 제목으로 바꿈).
> 요약·이후 기록은 [`progress_log.md`](progress_log.md).

---

## 2026-06-15

*원래 제목: 작업 일지 — 2026-06-15*

GRPO 파이프라인 실전 디버깅 → **LoRA 전환** → Stage-2 본실행 착수. 하루 동안 OOM·보상누락·형식붕괴·
처리량 병목을 차례로 규명하고, 클러스터 하드웨어 제약(NVLink 없음)에 맞춰 파이프라인을 **full-FT → LoRA**로
재설계했다.

---

### 1. GRPO 파일럿 OOM 연쇄 디버깅

- **pilot2(57188, full-FT/ZeRO-3)**: step 1 성공(MathAccuracy 0.0156>0, step_time ~9분)했으나
  **step 2에서 OOM**. 원인 정확 특정 — step 1의 `optimizer.step()`에서 Adam 상태(fp32)가 처음 할당되며,
  step 2 generation 시작 시 vLLM이 sleep→wake로 KV 캐시를 GPU에 재매핑(`cumem.py:241 create_and_map`)할
  자리가 없어 OOM. `vllm_gpu_memory_utilization 0.6`(48GB 예약)이 과함.
- **수정 시도**: num_generations 8→4, vllm util 0.6→0.4, **sleep_level 1→2**(학습 중 vLLM weight까지
  오프로드) → 새 에러 **`Cannot free a ZeRO-3 parameter while still active`**. **sleep_level 2 ↔ ZeRO-3 비호환** 확인.
- **최종 해결**: sleep_level 1 복귀 + **옵티마이저 상태 CPU 오프로드**(`configs/ds_zero3_offload.json`,
  ZeRO-3 네이티브라 충돌 없음). **pilot4(57201)**: 메모리 88→**66GiB**, **step 2(이전 OOM 지점) 통과** → OOM 해소 확정.

### 2. 🚨 MC(객관식) 정확도 누락 발견 & 해결

- 내장 `accuracy`(MathAccuracy/math_verify)는 `gold_parsed=parse(solution)`이 비면 reward=0으로 스킵.
  **객관식 letter("B")·기호("∠2")는 math_verify가 파싱 불가 → 정답이어도 항상 0점.**
- DeepVision 정답 분포(4000 표본): number 28% · **letter 48%** · expression(2√3+1) 21% · 기타 3%
  → **stage-2 보상 신호의 절반이 죽어 있었음** (파일럿 보상이 낮았던 진짜 원인).
- **해결**: 커스텀 `configs/accuracy.py`의 **`accuracy_mix`** ORM — gold가 수식이면 math_verify,
  letter(A~H)면 정규화 일치(B/(B)/"B. blue cube"/boxed 처리), 기타는 정규화 문자열 일치. **검증 9/9 통과.**
- `20_rlvr_grpo.slurm`에 `--external_plugins configs/accuracy.py --reward_funcs accuracy_mix format` 반영.
- **컨벤션 변경(`\boxed` 폐기)**: math_verify가 평문 숫자답을 정상 검증(`<answer>3</answer>`=1.0 확인)하므로
  `\boxed{}` 제거. `00_common.sh` SYSTEM_PROMPT를 `<think>간결추론</think><answer>답</answer>` + "concise"로 수정.

### 3. Format=0 진단 → format cold-start SFT

- pilot 출력(`log_completions`) 분석 — step 1의 32개 생성물: **`</think>` 닫음 5/32, `</answer>` 완결 0/32**,
  잘린 예시 7910자(2048토큰 꽉 참). 모델이 `<think>` 안에서 무한정 추론하다 결론 못 맺고 truncation.
  → **Format=0은 "형식 거부"가 아니라 "결론을 못 맺고 잘림"**(SFT 없는 베이스의 전형). throughput도 악화.
- **대응(1순위)**: 경량 format cold-start SFT. 데이터 = `UCSC-VLAA/VLAA-Thinking`의 **clevr_math**
  (시각-수학, answer가 이미 `<think>…</think><answer>X</answer>` 형식). `scripts/build_coldstart_sft.py`로
  추론 샌 장황건 제외 → **train 2913 / val 60** (`work/data/sft_coldstart_{train,val}.jsonl`).
  토큰 분포 확인: assistant median 220, full median 307 (간결 ✅).

### 4. 처리량 위기 → 근본원인: **NVLink 없음**

- cold-start SFT(full-FT) step이 **ZeRO-3 ~600초/step, ZeRO-2 ~375초/step** — 시퀀스 <1000토큰인데도 비정상.
- 배제: 긴 시퀀스(❌ <1000), 큰 이미지(❌ CLEVR 480×320), gradient_checkpointing(❌ 꺼도 느림).
- **`NCCL_DEBUG=INFO`**: `Channel 00 : 0[0]->1[1] via SHM` — 모든 GPU 간 통신이 **SHM(호스트 RAM 경유)** 폴백.
- **`nvidia-smi topo -m`**: GPU 전부 **PHB**(단일 CPU 호스트브리지), **NVLink(NV#) 없음**.
  GPU = **A100 80GB PCIe**, 노드 = OpenStack 가상화(`gpu-8-00x.novalocal`) → PCIe passthrough라 P2P 차단.
- **결론**: 8 GPU가 한 CPU 메모리로 몰려 경합 → **full-FT의 18GB gradient all-reduce가 통신 병목**.
  하이퍼바이저 ACS 영역이라 **우리가 못 고침**.

### 5. ✅ LoRA 전환 (전 단계)

통신량을 줄여 SHM 병목을 우회. base 동결 → adapter grad(수십 MB)만 통신.

- **cold-start SFT LoRA**(`10_sft.slurm TUNER_TYPE=lora`, DDP·DeepSpeed無, rank16/alpha32/all-linear, LR1e-4):
  138 step **12분** 완료, loss **0.89→0.46**. → `work/checkpoints/sft_coldstart_lora/.../checkpoint-138`.
- **병합**: `swift export --merge_lora` → `work/checkpoints/sft_coldstart_merged`(18GB full 모델).
- **GRPO LoRA 파일럿(57219, merged+lora)**:
  - **s/it 128초** (full-FT 662초 대비 **~5배↑**) → 70h에 ~1970 step 가능
  - **Format 0 → 0.156** (cold-start로 `<think></think><answer></answer>` 완결 시작!)
  - AccuracyMix 0.55(step1), 메모리 **59.6GiB OOM無**

### 6. 데이터셋 설계 결정

- medix 답은 "28x27mm"·"X-ray"식 **짧지만 규칙검증 brittle** → 의미적 LLM judge 필요.
- 한 GRPO run엔 동일 reward가 적용되므로 검증가능(DeepVision)+judge필요(medix) 혼합은 분기 보상 필요.
- **결정(원안 A)**: **Stage 2 = DeepVision(일반, accuracy_mix)만**, **Stage 3 = medix(judge)** 분리.
  망각 방지 위해 Stage 3는 medix + DeepVision 일부 혼합 예정.

### 7. Stage-2 GRPO LoRA 본실행 착수

- **job 57220** 제출 (`20_rlvr_grpo.slurm`, 기본 TUNER_TYPE=lora, num_gen 8, INIT_MODEL=sft_coldstart_merged,
  DeepVision, accuracy_mix+format, **70h 한도, save_steps 50**) → 산출 `work/checkpoints/grpo_general`.

---

### 신규/변경 파일
| 파일 | 내용 |
|---|---|
| `configs/accuracy.py` (신규) | `accuracy_mix` ORM — math_verify + letter + 문자열 |
| `configs/ds_zero2.json` (신규) | ZeRO-2 (full-FT SFT용, param all-gather 회피) |
| `configs/ds_zero3_offload.json` (신규) | ZeRO-3 + 옵티마이저 CPU 오프로드 (full-FT GRPO OOM 대비) |
| `scripts/build_coldstart_sft.py` (신규) | VLAA clevr_math → cold-start SFT jsonl |
| `scripts/00_common.sh` | SYSTEM_PROMPT `\boxed` 제거 + concise |
| `scripts/10_sft.slurm` | cold-start 데이터, TUNER_TYPE 분기(lora=DDP/full=ZeRO-2), max_length 2048, gc off |
| `scripts/20_rlvr_grpo.slurm` | accuracy_mix, TUNER_TYPE 분기(기본 lora), vllm util 0.4, mm_cache 0, log_completions |
| `scripts/30_medical_rl.slurm` | ds_zero3_offload, mm_cache 0 |

### 다음 할 일
- [ ] Stage-2(57220) reward/Format 추이 추적(체크포인트 50 step마다), throughput 재확인
- [ ] LLM-as-judge(`configs/medical_reward.py`) 실제 구현 — Stage 3 전제
- [ ] Stage 3: medix + DeepVision 일부 혼합 LoRA RL (judge 보상)
- [ ] num_generations·lora_rank 등 하이퍼파라미터 튜닝(메모리 여유 있음)

---

## 2026-06-16

*원래 제목: Worklog 2026-06-16 — 추론모드·길이 전략 정립*

- **추론모드 강제** — `enable_thinking=true` + `format_think` 보상(빈 `<think></think>`=0점, reward-hacking 차단).
- **flash_attn** 적용(sdpa→flash_attn). `vllm_util`↑는 무익(생성 병목·KV-bound 아님) 확인 후 원복.
- **soft_overlong** 길이 보상 추가(DAPO식). 단 "다 잘리면 그룹 내 분산=0"이라 보조 신호로 한정(가중치 0.2).
- **[핵심 진단] 추론 길이 폭주 규명** — base Qwen3.5-VL이 어려운 시각/기하 문제에 본래 3.5~4.6K토큰 추론.
  budget 늘려도(2048→6144) 50~80% 잘림(=무답=0점). `enable_thinking` on/off·콜드스타트(clevr_math 760자)와
  무관(일반화 실패). (`probe_coldstart_infer.slurm`)
- **[ZeRO-3 실험·불채택]** 길이 확대 위해 zero3+offload 시도 → 메모리 77.6→48GiB·8192 길이로 clip 0.5→0.25
  성공하나 **s/it 220→1,111(5배 느림)**. no-NVLink param all-gather 비용. 유효 학습량 LoRA-DDP 우세 → **불채택**.
- **[해결] rejection-sampling 간결 콜드스타트(STaR/ReST)** — 자기 롤아웃 정답+마감+간결 767예시(질문 336,
  p50 2.8K자)로 LoRA SFT→병합(`sft_rft_coldstart_merged`). **GRPO에서 FormatThink 0.05→0.27(5배)·clip↓ 검증**.
- **Stage-2 GRPO 본실행 시작**(job 57249) — LoRA-DDP + 6144 + RFT 간결 콜드스타트 init.
  보상 accuracy_mix+format_think+soft_overlong(1.0/0.2/0.2), flash_attn, s/it~202, mem 95GiB(안정).

---

## 2026-06-17

*원래 제목: Worklog 2026-06-17 — Stage-2 추세 확인 & 운영 정리*

- **Stage-2 GRPO 진행 중**(job 57249, ~17h·step ~300/38811). 50-step 구간평균 추세(노이즈 제거) — 일관 우상향,
  발산/붕괴 없음:

  | 구간 | Acc | FormatThink | clip(잘림) | mean_len | reward |
  |------|-----|-------------|-----------|----------|--------|
  | 1–50 | 0.419 | 0.233 | 0.434 | 3847 | 0.370 |
  | 101–150 | 0.420 | 0.322 | 0.376 | 3554 | 0.400 |
  | 201–250 | 0.432 | 0.349 | 0.377 | 3635 | 0.416 |
  | 251–300 | 0.436 | 0.409 | 0.357 | 3494 | 0.436 |

  → FormatThink 0.23→0.41(+76%), clip 0.43→0.36(잘림↓), mean_len 3847→3494, Acc 0.42→0.44.
  RFT 콜드스타트 효과가 학습 내내 유지·강화됨을 확인. (※ 전체 1000-step 추세는 worklog 06-19 참고)
- **목표 step**: full 3-epoch(38811 step=~90일)는 비현실 → RLVR plateau 도달 목적으로 **step 1000 부근 정지**
  (autostop 워처가 `checkpoint-1000` 저장 직후 자동 scancel). 도달 예상 ~2026-06-19.
- **[운영] 디스크 ~42G 회수** — 미사용 컨테이너(`ms-swift.sif`/`ms-swift-383.sif`)·대체된 콜드스타트
  (`sft_coldstart_merged`/`_lora`) 삭제, 스크립트 참조 정정(`build_image.sh`/`probe_*`/`00_common.sh`).

---

## 2026-06-19

*원래 제목: Worklog 2026-06-19 — Stage-2 baseline 완주 & plateau 진단*

- **Stage-2 GRPO baseline 완료**(job 57249) — step 1000 도달, autostop 워처가 `checkpoint-1000` 저장 직후
  자동 scancel. 총 2일 7.5시간(s/it~200). 산출: `grpo_general/v11-20260616-165537/checkpoint-{900,950,1000}`.
- **전체 1000-step 추세(100-step 구간평균)** — 발산/붕괴 없음, 일관 우상향:

  | 구간 | reward | Acc | FormatThink | clip | mean_len | zero_std |
  |------|--------|-----|-------------|------|----------|----------|
  | 1–100 | 0.377 | 0.418 | 0.259 | 0.420 | 3778 | 0.240 |
  | 301–400 | 0.444 | 0.435 | 0.443 | 0.349 | 3484 | 0.231 |
  | 501–600 | 0.534 | 0.500 | 0.525 | 0.307 | 3309 | 0.249 |
  | 901–1000 | 0.557 | 0.491 | 0.660 | 0.279 | 3267 | 0.328 |

- **[핵심 진단] Acc plateau** — reward(+48%)·FormatThink(+155%)·clip↓는 강하나 **Acc는 step~500서 0.50 정점
  후 0.47~0.50 정체**. 원인은 **`frac_reward_zero_std` 0.24→0.33 상승**: 학습 진행으로 그룹 내 rollout이
  "전부 정답/전부 오답"화 → 정확도 gradient 신호 소실. 후반 reward 상승은 형식/길이 주도(정확도 정체와 무모순).

---

## 2026-06-22

*원래 제목: Worklog 2026-06-22 — GRPO 파생기법 A/B 실험 착수*

- **[조사] ms-swift 4.1.3 GRPO 파생기법 인벤토리**(소스 직접 확인):
  - advantage_estimator: `grpo` / `rloo` / `reinforce_plus_plus`
  - loss_type: `grpo` / `bnpo` / `dr_grpo` / `dapo` / `cispo` / `sapo` / `real`
  - GSPO: `importance_sampling_level=sequence`(또는 `sequence_token`)
  - DAPO: `dynamic_sample`(+`max_resample_times`) / `epsilon_high`(clip-higher) / `delta`(two-sided) / `overlong_filter` / `soft_max_length`+`soft_cache_length`
  - reward scaling: `scale_rewards` = `group`/`batch`/`none`(Dr.GRPO)/`gdpo`
  - entropy: `top_entropy_quantile`(80/20) / `log_entropy`
  - off-policy 보정: `rollout_importance_sampling_mode`, `off_policy_sequence_mask_delta`, `async_generate`/`steps_per_generation`
  - 기타: CHORD(SFT+RL 공동), ref model 동기화, multi-turn, 내장 reward(accuracy/format/cosine/repetition/soft_overlong/math/react)
- **[실험] plateau 돌파용 A/B config** `scripts/21_rlvr_grpo_adv.slurm` — baseline과 **기법 외 전 조건 동일**.
  공통 코어 = **`dynamic_sample`**(zero_std 그룹 폐기·재샘플, 진단 직격) + `overlong_filter`. 2개 일관 레시피:
  - `dapo`(기본): clip-higher(ε 0.2/ε_high 0.28) + `loss_type=dapo` (토큰레벨)
  - `gspo`: `importance_sampling_level=sequence` + 시퀀스레벨 작은 clip(3e-4/4e-4)
- **스모크 테스트 통과**(job 57526, RECIPE=dapo `--max_steps 5`, COMPLETED) — 신규 인자(dapo/dynamic_sample/
  overlong_filter/epsilon_high/log_entropy) 전부 수용. step_time 182/197/201/143/160s(평균 ~177s, baseline 동급).
  step 1~2 `frac_reward_zero_std=0`(dynamic_sample 작동), entropy 로깅 정상, mem 62GiB. 산출 `grpo_general_adv_dapo/`.
- **dapo 본실행 제출**(job 57527, `--max_steps 1000`, baseline와 동일 지점·init·보상 → clean A/B).
  DAPO 4대 기법(dynamic_sample/clip-higher/loss_type=dapo/overlong_filter) 적용. README "GRPO 파생기법" 절 참고.
- 판정기준: `frac_reward_zero_std`↓ + AccuracyMix↑(0.49 돌파) + entropy 비붕괴.

---

## 2026-06-24

*원래 제목: Worklog 2026-06-24*

Stage-2 GRPO 파생기법 A/B — **DAPO 본실행(job 57527) 진행 모니터링** (step 408→475/1000).

### 1. 학습 안정성 정밀 분석 (baseline vs DAPO)
로그(`logging.jsonl`)에서 stability 지표를 직접 산출해 "이전 방법론(baseline GRPO 57249) 대비 더 안정적인가" 정량 검증.

| 안정성 지표 | baseline(57249) | DAPO(57527) | 판정 |
|---|---|---|---|
| grad_norm 평균 | 2.14(동일구간) / 671(전체) | **0.012** | DAPO 압도 |
| grad_norm 최대 | 681(동일구간) / **67만**(전체) | **0.024** | 폭주 사실상 0 |
| grad spike(>5×median & >1.0) | 9/403(동일구간), 107/1000(전체) | **0/403** | 무폭주 |
| frac_reward_zero_std | 0.26 (max 0.69) | **0.000** | plateau 원인 제거 |
| reward series std | 0.161 | **0.119** | 더 평탄 |
| kl | 0.036 | 0.069 | baseline 약간 낮음(통제범위) |

- ✅ **안정성은 논쟁의 여지 없이 DAPO 우세**: grad 폭주 제거 + zero_std 0 완전유지 + reward 평탄화.
- baseline 전체 1000-step 에서 grad_norm 이 최대 **67만**까지 튀는 심각한 불안정 → DAPO 는 403step 내내 0.012 수준 무spike.

### 2. ⚠️ Acc 약한 적신호 (step 475 기준)
- 구간추세: Acc 1–100 **0.479** → 301–400 0.436 → 최근50 **0.427** (완만한 하락).
- FormatThink 0.64 포화됐는데 정확도 미동반 → **형식은 잡혔으나 정답률 정체/소폭 하락**.
- KL 0.048→0.089 완만 상승 → clip-higher(ε_high 0.28)가 정책을 형식 쪽으로 밀며 정확도 신호 일부 희생 가능성.
- 🎯 **분수령 = step 600 돌파 구간**: baseline 은 step ~600 에서 Acc 0.50 으로 점프. DAPO 가 그 구간서 따라잡는지가 "안정성→정확도 전환" 여부 확정.

### 3. step 600 돌파 판정 자동화
누적평균(step 1~N) A/B 표가 baseline 의 step ~600 Acc 도약을 희석하는 문제 해결.
- `scripts/grpo_ab_update.py` 보강: `mean_window()` 추가, **N≥600 시 step 501~600 구간 Acc 를 baseline vs DAPO 직접 비교**해 `돌파 확인/미확인/동률` 판정을 README A/B 블록에 자동 출력.
- 두 분기(N<600 / N≥600) 임시 README dry-run 검증 완료. watcher 가 working-tree 스크립트를 직접 실행하므로 즉시 라이브.
- 커밋 `60f74eb`.

### 4. 진행 갱신
- watcher(`grpo_watch.sh`, PID 118214) 25h+ 정상 상주. step 400 까지 자동갱신·push 완료.
- step 475 시점 README A/B 표 + 비교 plot 수동 재생성·갱신·push (커밋 `548a220`).
- ETA: step 600 ≈ 06-25 새벽(13h), step 1000 완주 ≈ 06-26~27 (370s/it). 잡 시간제한 21h 잔여 → step 600 도달 후 시간제한 도래 가능, 재큐 또는 checkpoint 재개 필요.

---

## 2026-06-25

*원래 제목: Worklog 2026-06-25*

DAPO A/B 종결 → 추가 GRPO 파생기법 조사 → **dr_grpo 본실행 착수**.

### 1. DAPO(57527) 종결 — 돌파 미확인
- step 600 자동 돌파판정(어제 구축) 정상 작동: **step 501~600 Acc DAPO 0.465 < baseline 0.500 (Δ-0.035)** → "안정성 이득이 정확도로 미전환".
- step 623 시점 `scancel` 으로 종결, **`checkpoint-600` 을 산출물로 확정**.
- 종합 결론: **안정성은 baseline 압도**(grad_norm 무폭주·`frac_reward_zero_std` 0.00 유지)하나 **정확도 우위 미확보**.
  추정 원인 = 후반 길이↑(3380→3620)→clip↑(0.285→0.32) 재폭주로 잘린 응답이 0점화 → Acc 정체.

### 2. ms-swift 4.1.3 GRPO 파생기법 인벤토리 (컨테이너 소스 직접 검증)
`rlhf_trainers/args_mixin.py` + `grpo_trainer.py` 분기 기준 — 실제 구현·지원 확인.

**loss_type** (집계/클리핑, advantage=GRPO 유지): `grpo`·`dapo`·`bnpo`·`dr_grpo`·`cispo`(MiniMax M1)·`sapo`·`real`·`discopop`.
**advantage_estimator**: `grpo`·`rloo`(leave-one-out)·`reinforce_plus_plus`.
**importance_sampling_level**: `token`·`sequence`(GSPO)·`sequence_token`(GSPO-token).
**직교 노브**: `top_entropy_quantile`(고엔트로피 분기토큰만 학습)·`delta`(dual-clip)·`num_iterations`(K>1 다중업데이트)·
`sync_ref_model`+`ref_model_sync_steps`·`scale_rewards group|batch|none|gdpo`·`whiten_rewards`·`kl_in_reward`.

### 3. dr_grpo 본실행 착수 (job 57624, gpu-8-001)
DAPO 진단(길이 폭주)을 직격하는 **Dr.GRPO**(arXiv:2503.20783) 선택 — 두 편향 제거:
- `loss_type=dr_grpo`: 손실을 시퀀스길이가 아닌 **상수(batch×max_completion)로 정규화** → 길이정규화 편향 제거
  (소스: `per_token_loss.sum()/(B*max_len)`).
- `scale_rewards=none`: advantage 를 그룹 std 로 나누지 않음 → 난이도(쉬운 문제 과대가중) 편향 제거
  (소스: `scale_rewards!='none'` 일 때만 `adv/(std+1e-4)`).
- 공통 코어(dynamic_sample + overlong_filter) 유지, clip-higher 미적용(대칭 ε 0.2)으로 dr_grpo 효과만 분리.
- init = `sft_rft_coldstart_merged`(baseline·DAPO 와 동일) → clean A/B. 출력 `work/checkpoints/grpo_general_adv_dr_grpo`.
- 스크립트: `scripts/21_rlvr_grpo_adv.slurm RECIPE=dr_grpo` (레시피 추가 반영). 70h 타임리밋 풀확보.

### 4. 모니터링
- DAPO 전용 watcher(57527 고정) 종료. README AUTO 블록은 DAPO 확정기록으로 보존.
- dr_grpo 추적 파이프라인(watcher/updater 일반화)은 별도 구성 예정.

---

## 2026-06-28

*원래 제목: Worklog 2026-06-28*

**Stage-2 종결 — dr_grpo 돌파 확인.** GRPO 파생기법 A/B 의 최종 결론.

### 1. dr_grpo(57624) 종료 + 돌파판정
- step 689 에서 **TIMEOUT 정상 종료**(2026-06-28 06:08, max_steps 미설정이라 시간제한까지 진행).
- **돌파판정 = ✅ 돌파 확인**: step 501~600 구간 Acc **dr_grpo 0.526 > baseline 0.500** (Δ+0.027).
  → DAPO 가 0.465 로 미끄러진 바로 그 구간을 **통과**.
- 동반 지표: 같은 구간 **mean_len 3259**(baseline 3309·DAPO 후반 ~3600 대비 짧음) → "길이 억제 + Acc 상승" 동시 달성.

### 2. 가설 최종 검증
이 실험의 가설 — *"DAPO 의 정확도 미전환은 길이 폭주 때문, 길이편향을 제거하면 풀린다"* — 가 입증됨:
| 기법 | step501~600 Acc | mean_len 궤적 | 판정 |
|------|-----------------|---------------|------|
| GRPO(baseline) | 0.500(정점 후 정체) | — | plateau |
| DAPO | 0.465 | 3380→3600 ↑ | ❌ 돌파 미확인 |
| **dr_grpo** | **0.526** | **3259 유지/↓** | ✅ **돌파 확인** |

### 3. 후반 꼬리(601~689) 분석 → checkpoint-600 최적
- Acc 0.526(501~600) → **0.494**(601~689) 소폭 냉각, mean_len 3259→3426 반등.
- 단 발산 아님: 전체 689 step NaN 0 · grad_norm spike 0 · zero_std 0.00 유지.
- 해석: 돌파 정점이 step 600 부근 → **산출물 = `checkpoint-600`** 으로 확정(꼬리 구간 학습은 이득 없음).

### 4. Stage-2 최종 결론
- **승자 = dr_grpo `work/checkpoints/grpo_general_adv_dr_grpo/v0-20260625-081142/checkpoint-600`**.
- 3기법 비교 완결: GRPO(정체) → DAPO(안정하나 길이폭주로 실패) → **dr_grpo(편향 제거로 돌파)**.
- README A/B 블록·plot·status 최종 갱신. dr_grpo watcher 종료.

### 5. 다음 (Stage-3)
- dr_grpo `checkpoint-600` 을 init 으로 **의료 특화 RL**(LLM judge) 착수.
- `configs/medical_reward.py` judge 구현이 선행 과제.

---

## 2026-06-29

*원래 제목: Worklog 2026-06-29*

**Stage-3 착수 — RaR(Rubric-as-a-Reward) 루브릭 보상 설계·구현.**

### 1. 자료조사: Rubric-as-a-Reward (RaR)
- RaR([arXiv:2507.17746](https://arxiv.org/abs/2507.17746), Scale AI): 비검증 도메인에 RLVR 확장. 체크리스트형 루브릭을
  judge 가 항목별 0/1 → explicit 집계 `r=Σwⱼcⱼ/Σwⱼ`. HealthBench 최대 +28~31%.
- 4대 원칙: 전문가지식 기반 / 포괄 커버리지 / 의미적 가중 / 자기완결. 카테고리 가중 Essential 5·Important 3~4·Pitfall 4(정수).
- 의료 사례(Baichuan-M2 등): 정확성·근거·안전성·환각 등 다차원, positive/negative 루브릭.

### 2. RaR-Medicine-20k 활용 검토 → 비채택(스키마만 차용)
- `anisha2102/RaR-Medicine`(HF, 비게이팅, 22.4k): `{question, reference_answer, rubric[{title,description,weight}], rubric_count(5~17)}`.
- 실제 샘플 확인(val 3건): medical-o1 등 **텍스트전용·장문 임상추론**, 9~12 인스턴스 루브릭(GPT-4o 생성).
- ❌ 우리 과제(멀티모달 단답 VQA)와 성격 불일치 → 학습데이터 비채택. ✅ 루브릭 **스키마·few-shot 참고용**만.

### 3. 데이터 실측 → 루브릭 사이징 결정
- medix(`medix_rl_train.jsonl`) = **단답 의료 VQA**: 정답 중앙값 46자(예 "<CT_SCAN> 28 x 27 mm", "<X_RAY> X-ray").
- 단답이라 RaR식 7~20항목 인스턴스 루브릭은 과설계 → **정적 컴팩트 4차원**이 더 풍부한 신호.
- 핵심사실=참조답 1개 → **참조답을 Essential 기준에 템플릿 주입** → 오프라인 LLM 루브릭 생성 파이프라인 불필요.

### 4. 확정 루브릭 (spec §4.2)
| # | title | weight | 대상 |
|---|-------|--------|------|
| c1 | 정답 정확성(참조답 주입) | 5 Essential | `<answer>` |
| c2 | 시각근거(영상 소견 인용) | 3 Important | `<think>` ⭐교차추론 |
| c3 | 정밀도·단위(±15%) | 3 Important | 측정형 자동분기 |
| c4 | 환각·과잉주장 없음 | 4 Pitfall | 전체 |
- explicit 집계, 형식위반→judge생략 0, 측정형(`\d+\s*(mm|cm|...)`) 자동 c3 포함.

### 5. 구현 + 검증
- `configs/medical_reward.py` `ClinicalJudgeReward(AsyncORM)`: 형식게이트→멀티모달 judge API(env `JUDGE_BASE_URL/MODEL/API_KEY`)
  →JSON 0/1 파싱→explicit 집계. 이미지 base64 멀티모달 전달. 타임아웃·네트워크·파싱실패→0.0.
- `scripts/test_medical_reward.py`: `swift.rewards` 스텁 + mock judge → **24/24 통과**(측정형 감지·루브릭구성·형식게이트·
  JSON파싱·집계 단조성 1.0>0.8>0.0·async mock).

### 6. judge 모델 확정·검증 (Qwen3.6-27B-FP8)
- 후보 검토: snuhub/OpenRouter(외부 API) → **컴퓨트노드 오프라인이라 학습중 도달 불가**(rate-limit·거버넌스도 부적합) → **내부 self-host** 결론.
- 로그인노드 self-host 시도 → 🚨 **불가**: 로그인노드 드라이버 **470(CUDA11.4)** 이라 컨테이너 vLLM(CUDA12.9 빌드) 로드 실패
  (`undefined symbol: cuTensorMapEncodeTiled`). GPU(A100 40GB) 있어도 못 씀. → **judge는 컴퓨트노드(드라이버550)** 확정.
- judge = **`Qwen/Qwen3.6-27B-FP8`**(Apache-2.0, 비게이팅, ~31GB): config 실측 `model_type=qwen3_5`·`Qwen3_5ForConditionalGeneration`
  = **우리 base 와 동일 arch** → 컨테이너 vLLM 0.19.1 그대로 서빙(Gemma4식 CUDA13 함정 회피). `vision_config` 존재 → 멀티모달.
  - (GGUF 5~6bit 경로도 검토: mmproj 있어 멀티모달 가능하나 llama.cpp 별도 빌드 필요 → FP8+vLLM 우선.)
- **스모크 검증**(`scripts/judge_server.sh` + `31_judge_smoke.slurm`, 1gpu 40GB):
  - ✅ FP8 30GB **단일 40GB 적합**(36.6GB 사용, OOM 없음, ready ~300s).
  - ✅ 멀티모달 채점(이미지 전달) + thinking off → **파싱가능 JSON** `{"c1":1,...}`.
  - ✅ 실제 `medical_reward` 경로 end-to-end: **정답 1.0 > 오답 0.0 단조성 PASS**.
  - 부수수정: Qwen3 추론모델이라 기본 응답 content 비어 파싱 0 → `enable_thinking=False`(chat_template_kwargs) + reasoning_content 폴백 추가.
- 40GB 보수설정: `--max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image":1}'`(vLLM0.19.1은 JSON 문법).

### 7. 남은 선행과제 (Stage-3 본실행 전)
- 학습↔judge **내부망 도달성 테스트**(학습노드↔judge노드 IP:port, 별도 잡 2개).
- spec §8 **분포 프로브**: medix 100~200건 judge 점수 분포·**c2(시각근거) 캘리브레이션**(스모크에선 다소 관대).
- `30_medical_rl.slurm` 배선: init=dr_grpo `checkpoint-600`, `--reward_funcs format_think clinical_judge`, judge 노드 env(`JUDGE_BASE_URL` 등) 자동 주입.

### 7. 기타
- ⚠️ 보안: `~/model_download.py` 에 HF 토큰 평문 노출 발견 → 재발급·env 분리 권고.
- README 정리(현황 날짜·종결 표기·진행이력 중복 제거·Stage-3 섹션 추가).

### 8. Stage-2 closeout (merge + 학습효과 벤치마크)
- dr_grpo `checkpoint-600`(LoRA) → `dr_grpo_merged`(18G) 병합(`merge_drgrpo.slurm`). Stage-3 init 겸.
- **base vs 학습모델 비교**(`40_eval_compare.slurm`+`eval_compare.py`, DeepVision 홀드아웃 100건, accuracy_mix):
  | 지표 | base(Qwen3.5-9B) | 학습모델 | 변화 |
  |---|---|---|---|
  | 정확도 | 0.21 | **0.35** | +67% 상대 |
  | 형식(`<answer>`) | 0.23 | **0.46** | 2배 |
  | 평균길이(자) | 5618 | 4414 | -21% |
  → 콜드스타트+GRPO 가 정확도·형식·간결성 모두 개선. 정량 입증.
  - 함정: 컨테이너는 HF id 를 modelscope 로 받으려다 read-only(/mnt/workspace) 실패 → **로컬 스냅샷 경로**로 서빙해야 함.
  - base 는 형식 미학습이라 max_tokens 까지 장황 생성(느림) → conc↑·max_tokens 2048 로 평가.
- Stage-3 배선 완료(`30_medical_rl.slurm`·`judge_server.slurm`·`launch_stage3.sh`). 다음 = `bash scripts/launch_stage3.sh` 본실행.

---

## 2026-07-01

*원래 제목: worklog 2026-07-01*

### 요약
Stage-3(RaR) 배선을 **GRPO 루프 안 end-to-end 스모크**로 실증. 정적 검증이 못 잡는 **이미지 형식 실버그**를 발견·수정하고 재검증 PASS. Stage-2 1-epoch 재학습은 계속 진행 중.

### 배경 (직전 6-29~30)
- 정적 vs 인스턴스 루브릭 실증 비교 → **정적 통일 루브릭 채택**(환각변별 +0.338 vs +0.021).
- 평가 누수 발견 → 정답유형 **층화 홀드아웃 분리**(`make_holdout.py`: math 453 + vl 519 = 972, trainonly 102,531) + **init 부터 fresh 1 epoch 재학습**(job 58892~, MAX_STEPS=3204). 이전 +67% 벤치마크 무효화.

### 오늘 한 것

### 1. Stage-3 배선 정적 검증 (실제 swift 4.1.3 소스 대조)
`from swift.rewards import AsyncORM, orms` 존재, 6개 계약 전부 일치 확인:
- `orms[reward_func]` 조회 → `reward_func_class(args=args)` 인스턴스화 (grpo_trainer.py:2279-2280)
- AsyncORM 감지 → `asyncio.gather` 병렬, `await func(completions, **reward_kwargs)` (381)
- `reward_kwargs.update(RowPreprocessor.rows_to_batched(reward_inputs))` (350) → 데이터 컬럼(solution/images) 배치 주입

### 2. end-to-end 스모크 (`35_stage3_smoke.slurm`)
- idle 8gpu 노드(gpu-8-003)에서 GPU0=judge(Qwen3.6-27B-FP8) / GPU1=소형 GRPO 트레이너(max_steps 2, num_gen 2).
- `medical_reward.py` 에 `JUDGE_DEBUG` 훅 추가 — 첫 호출에 images kwarg 구조·경로·data_url 여부 로깅.
- 스모크용 소형 데이터 `medix_smoke.jsonl`(64줄).

### 3. 🐛 실버그 발견·수정 — swift 는 `images` 를 dict 로 넘김
```
[JUDGE_DEBUG] images[0] type=list first_img type=dict value="{'bytes': None, 'path': '.../medix-rl-data_2_0.png'}"
[JUDGE_DEBUG] path_exists=N/A data_url_ok=False (→ 시각근거 c2 채점에 이미지 누락)
```
- 원인: swift GRPO 가 `images` 를 str 경로가 아니라 HF datasets 포맷 dict `{'bytes':..,'path':..}` 로 전달. 기존 `_image_to_data_url` 은 str 만 처리 → **이미지 누락 → 시각근거(c2) 항상 blind**.
- 정적 검증(소스/유닛-스텁)으로는 못 잡음 — 실제 데이터 흐름에서만 드러남. 스모크의 존재 이유.
- 수정: `_image_to_data_url` 이 **str 경로 / dict{bytes,path} / PIL.Image** 모두 처리.
- 회귀 방지: 유닛테스트 +5(이미지 형식) → **29/29 통과**.

### 4. 재검증 (수정본) — PASS
```
[JUDGE_DEBUG] resolved_path='.../medix-rl-data_2_0.png' path_exists=True data_url_ok=True (→ 이미지 전달됨)
Step1: FormatThink=1.0  ClinicalJudgeReward=0.5833  reward=0.7833
Step2: FormatThink=1.0  ClinicalJudgeReward=1.0     reward=1.2
트레이너 종료코드=0
```
- 이미지 judge 도달 ✅, RaR judge 실채점(all-0 아님 = 시각근거 포함) ✅, 보상 통합 `0.2·Format + 1.0·Clinical` 정확 일치 ✅.

### 검증된 것 / 남은 것
- ✅ Stage-3 배선 end-to-end 동작 실증(플러그인 로드·clinical_judge 해석·kwargs 도달·이미지 도달·보상 통합).
- ⏳ Stage-2 1-epoch 완주 → 최종 ckpt 재병합(`dr_grpo_merged` 갱신) → 층화 홀드아웃 벤치마크(math/vl 분리) → Stage-3 init 교체 → `launch_stage3.sh` 본실행.

### 커밋
- `ce9b177` fix: RaR judge 이미지 형식 버그(dict{bytes,path}) + 스모크 스크립트 + 유닛 29/29
- `b62d427` docs: HANDOFF 스모크 결과·images dict 함정 기록
