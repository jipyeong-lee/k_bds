# HANDOFF — K-BDS 의료 멀티모달 교차추론 학습 파이프라인

> 작성: 2026-06-23 / 다음 작업자(또는 다음 세션)를 위한 인수인계.
> 상세 배경은 `README.md`(현황·의사결정 이력)와 `docs/worklog_*.md`(일별) 참조.

---

## 1. 지금 무슨 일이 돌고 있나 (TL;DR)

- **Stage-2 범용 RLVR/GRPO** 단계. baseline 완주 후 **DAPO A/B 실험 진행 중**.
- **실행 중인 잡: `57527` (grpo-adv, DAPO 레시피)** — RUNNING, `--max_steps 1000`, 현재 **step ~179/1000**, ~367 s/it.
  - 제출: 2026-06-22 15:39 / 도달 예상: step 1000 ≈ **2026-06-26~27**.
  - `max_steps=1000`이 박혀 있어 **워처 없이 1000에서 자동 종료**됨.
- **자동 README 갱신 모니터 가동 중** (하네스 Monitor task `buwjpegbg`, persistent):
  새 `checkpoint-N00`(100의 배수) 저장 또는 잡 종료 시 → 추세표·plot·README 갱신 후 커밋·푸시.
  ⚠️ **세션이 끊기면 이 모니터도 멈춤** → 재개 시 누락분 일괄 반영 필요(아래 §6 절차).

---

## 2. 핵심 잡 2개 (A/B)

| | baseline | DAPO (현재 실행) |
|---|---|---|
| Job ID | 57249 (CANCELLED@step1000, 정상완주) | **57527 (RUNNING)** |
| 스크립트 | `scripts/20_rlvr_grpo.slurm` | `scripts/21_rlvr_grpo_adv.slurm` (RECIPE=dapo) |
| 로그 | `logs/grpo_stage2_57249.log` | `logs/grpo_adv_57527.log` |
| 출력 | `work/checkpoints/grpo_general/v11-20260616-165537/` | `work/checkpoints/grpo_general_adv_dapo/v1-20260622-154040/` |
| init 모델 | `sft_rft_coldstart_merged` | `sft_rft_coldstart_merged` (동일) |
| 결과 | step1000 완주, **Acc plateau** | step~179, **frac_zero_std 0.24→0.00**, FormatThink 수렴 2배 |

**DAPO 추가 인자(baseline 대비 차이, 그 외 전 조건 동일):**
`--dynamic_sample true --max_resample_times 3 --overlong_filter true --loss_type dapo --epsilon 0.2 --epsilon_high 0.28 --importance_sampling_level token`
(KL `beta=0.04`는 유지 — DAPO 원논문 0과 다름.)

**A/B 중간 결과 (step 1~176 동일구간):** frac_zero_std 0.24→**0.00**(dynamic_sample 가설 검증, plateau 직격),
FormatThink 0.29→0.48, reward 0.39→0.48, clip 0.40→0.31. 단 **~1.8배 느림**, **Acc 이득은 미확정**(추가 step 후 재판정).

---

## 3. 환경 / 실행 방법

- **클러스터**: KISTI K-BDS Slurm. `8gpu` 파티션 = **A100 80GB PCIe ×8, NVLink 없음**(가상화·SHM 폴백).
  → full-FT 멀티GPU는 통신병목(375~660s/step) → **전 단계 LoRA로 수행**(~5배 빠름). 이게 핵심 제약.
- **컨테이너**: Singularity sandbox `work/images/ms-swift-413-sandbox` (swift4.1.3 / torch2.10 / vllm0.19.1 / flash_attn2.8.3).
  - 재빌드 원본: `work/images/ms-swift-413.sif` (+ `env/build_image.sh`). 구 폴백 이미지는 디스크 정리로 삭제됨.
  - glibc 2.17이라 conda로 vLLM 불가 → 컨테이너가 정식 경로.
- **공통 설정**: `scripts/00_common.sh` (WORK_DIR, BASE_MODEL, CONTAINER_IMG, SYSTEM_PROMPT, NPROC_PER_NODE=8 기본).
  - ⚠️ 단일 GPU 디버그 잡은 `export NPROC_PER_NODE=1` 필수(안 하면 8프로세스 launch→CUDA invalid device).
- **GRPO 제출 패턴**:
  ```bash
  sbatch --job-name=grpo-adv \
    --output=logs/grpo_adv_%j.log --error=logs/grpo_adv_%j.log \
    scripts/21_rlvr_grpo_adv.slurm        # RECIPE=dapo 기본, RECIPE=gspo 토글 가능
  ```
- **푸시**: `git push origin HEAD:master` (로컬 브랜치 main → 원격 master).

---

## 4. 보상 설계 (Stage-2)

`configs/accuracy.py` (external_plugins로 로드), `--reward_funcs accuracy_mix format_think soft_overlong`, 가중치 `1.0 0.2 0.2`.
- **`accuracy_mix`** — 객관식 letter 정답 파싱(내장 accuracy의 치명 누락 보완).
- **`format_think`** (`FormatThink` ORM) — `<think>…</think><answer>…</answer>` 닫힌 형식 + think ≥16자 요구
  (빈 `<think></think>`=0점 → reward-hacking 차단).
- **`soft_overlong`** — DAPO식 길이 보상. 단 "다 잘리면 그룹 내 분산=0"이라 보조 신호로만(가중치 0.2).

---

## 5. 데이터 / 모델 자산

- **데이터** (`work/data/`): `deepvision103k_train.jsonl`(103K, Stage-2 범용), `medix_rl_train.jsonl`(51K, Stage-3 의료),
  `sft_rft_coldstart_{train,val}.jsonl`(727/40, rejection-sampling 콜드스타트 SFT용).
- **모델 체크포인트** (`work/checkpoints/`):
  - `sft_rft_coldstart_merged` (18G) — **현재 모든 GRPO의 init**. (구 `sft_coldstart_merged`는 디스크 정리로 삭제됨)
  - `grpo_general/v11-.../checkpoint-1000` — baseline 최종 LoRA 어댑터.
  - `grpo_general_adv_dapo/v1-.../checkpoint-{50,100,150,...}` — DAPO 진행분.
- **콜드스타트 생성**: `scripts/build_rft_coldstart.py` (자기 롤아웃 정답+마감+간결 필터 → SFT 데이터).

---

## 6. 진행 중 작업 / 다음 할 일

### 즉시 (모니터가 자동 처리하도록 설정됨)
- [▶] **100-step마다 README 갱신** (Monitor `buwjpegbg`). 세션 끊겨 누락 시 수동 절차:
  1. 추세 재계산: 아래 §7 plot/표 명령 실행
  2. plot 재생성: `singularity exec work/images/ms-swift-413-sandbox python scripts/plot_grpo_compare.py logs/grpo_stage2_57249.log baseline logs/grpo_adv_57527.log DAPO docs/assets/grpo_dapo_vs_baseline.png`
  3. README 현황·A/B 표 갱신 → `git add -A && git commit && git push origin HEAD:master`

### DAPO 완주 후
- [ ] **A/B 최종 판정**: step 1000까지 baseline vs DAPO 비교 → plateau(Acc) 돌파 여부 확정.
- [ ] (선택) `RECIPE=gspo`(sequence-level IS) 레시피도 동일 스크립트로 비교.
- [ ] 승자 체크포인트 merge + 추론 검증 (`scripts/merge_probe_rft.slurm` 패턴 재사용).

### Stage-3 (의료 RL) — 미착수
- [ ] `configs/medical_reward.py` **LLM-as-judge 실제 구현** (현재 스켈레톤, 스펙: `docs/medical_reward_spec.md`).
- [ ] `scripts/30_medical_rl.slurm`로 medix + DeepVision 일부 혼합 LoRA RL (망각 방지).
- [ ] `scripts/40_eval.slurm`의 EVAL_DATASETS를 **실제 의료 멀티모달 벤치마크**로 교체.

---

## 7. 자주 쓰는 명령 (모니터링/분석)

```bash
# 잡 상태
squeue -u k252a01 -o "%.10i %.20j %T %.10M %R"

# 최신 step / 속도
grep -oE "'global_step/max_steps': '[0-9]+/[0-9]+'|'train_speed\(s/it\)': '[0-9.]+'" logs/grpo_adv_57527.log | tail -2

# 적용 인자 확인 (args.json)
python3 -c "import json;a=json.load(open('work/checkpoints/grpo_general_adv_dapo/v1-20260622-154040/checkpoint-150/args.json'));print({k:a[k] for k in ['epsilon_high','dynamic_sample','overlong_filter','loss_type','beta']})"

# baseline vs DAPO 구간평균 비교 (step 1~현재) — scripts/plot_grpo_compare.py 와 동일 파싱 로직
# (추세표는 대화 중 python 스니펫으로 100/50-step bin 평균 계산)
```

**핵심 지표:** `frac_reward_zero_std`(↓=학습신호 효율), `rewards/FormatThink/mean`, `rewards/AccuracyMix/mean`,
`completions/clipped_ratio`(잘림=무답), `completions/mean_length`.

---

## 8. 함정 / 주의사항 (Lessons learned)

- **NVLink 없음** → ZeRO-3/full-FT 멀티GPU는 param all-gather로 5배 느림(테스트 완료, 불채택). **LoRA-DDP만 사용**.
- **추론 길이 폭주**: base Qwen3.5-VL이 어려운 문제에 본래 3.5~4.6K토큰 장문 추론 → 잘림=0점이 학습 정체 원인.
  해결책은 budget 확대(메모리/속도 천장)가 아니라 **rejection-sampling 간결 콜드스타트**.
- **단일 step 지표는 노이즈 큼** → 반드시 50/100-step **구간평균**으로 판단.
- **스모크(5 step) 속도 ≠ 본실행 속도**: dynamic_sample 재샘플 부하가 초기엔 작아 과소추정됨(177s→실제 367s/it).
- **DAPO ≠ soft_overlong만**: DAPO는 dynamic_sample+clip-higher+token-level loss+overlong_filter 4종 묶음.
  baseline(57249)은 soft_overlong 보상 1개만 차용한 표준 GRPO였음(혼동 주의).
- `mem 86~95 GiB`는 `max_memory_reserved` watermark(안정값)이지 OOM 아님.

---

## 9. 참고 위치

- **README.md** — 현황(날짜별)·환경·학습방식·보상·DAPO 레시피·자원·데이터 전체.
- **docs/worklog_2026-06-{15,16,17,19,22}.md** — 일별 상세.
- **docs/medical_reward_spec.md** — Stage-3 judge 스펙(구현 전).
- **docs/assets/grpo57249_trend.png** — baseline 단독 추세.
- **docs/assets/grpo_dapo_vs_baseline.png** — baseline vs DAPO 비교(점선=DAPO).
- **메모리**: `~/.claude/.../memory/` (클러스터 환경·과제목표·glibc/컨테이너 제약).
