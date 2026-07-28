# 운영 · 데이터 · 홀드아웃

> 이 문서는 [`README.md`](../README.md) 에서 분리된 상세 기록입니다. 요약·현황은 README, 상세는 여기.

## 운영 · 데이터

### 자원 & 운영 정책 (가이드)
- 계획서 **8×A100-80GB 노드** = `8gpu` 파티션(4gpu·8gpu=80GB, 1gpu·2gpu=40GB). `bdata_user` 바로 사용, diba 불필요. **80GB 실측 확인**(파일럿 59.6GiB).
- **wall-clock 5일(120h)**: 70h 단일 OK(`--resume_from_checkpoint` 권장). **배타적 노드**(1노드=1작업). **동시 제출 8gpu 최대 6개**(노드 3×2). **노드시간** 8gpu=8/h(`cat /scratch/account/kbds0754`). `#SBATCH --comment=pytorch` 필수.

#### 🔴 예산 실적 (2026-07-21 기준, `sacct` 실측)

**4,164.4 / 5,000 노드시간 소진 (83.3%) → 잔여 ~836.** 계획서 방식(8gpu = ×8)으로 계산.

| 실측 항목 | 노드시간 |
|---|---|
| 8gpu 잡 40개 (Stage-2 GRPO 계열 대부분) | 4,112.3 |
| 그 외(4gpu 45.8·1gpu 3.2·debug 2.6·2gpu 0.4·cpu 0.1) | 52.1 |
| **Stage-2 GRPO 1회** (70h×8) | **490~560** ← k252a02(5,000h)서 실행 중. 1 epoch≈2,337 step 은 체인 4잡 ≈**1,719** |
| **SFT 1회** (12~39분) | **2~5** ← 사실상 공짜 |
| Stage-3 (미시작, 계획서 핵심 산출물) | ~500 예상 |

> **2026-07-22 예산 방향전환 → 07-27 이관 완료**: k252a02 에 5,000 노드시간 확보 → Stage-2 풀확장 + Stage-3 둘 다 가능. **현재 이 레포가 그 계정**(work/ 293G 전송·경로 치환 완료). → [Stage-2 풀확장](stage2_experiments.md#0-풀확장-재설계-2026-07-2224-데이터--07-28-본실행-착수) · [`HANDOFF.md`](../HANDOFF.md)

→ **콜드스타트 v3 재구축은 저렴(SFT+평가 ~15)하나, 그 위에 Stage-2 를 다시 돌릴 여력은 없다.** 잔여 배분은 v3 SFT+평가 수치를 보고 결정.
*(Slurm 에 하드 리밋은 없음 — `GrpTRESMins` 미설정이라 잡이 거부되진 않는다. 계획서/보고 기준.)*

<details><summary>계획서 원안 (4,960 / Track III 한도 5,000)</summary>

| 단계 | 스크립트 | 1회 | 반복 | 노드시간 |
|------|----------|-----|------|----------|
| SFT | 10_sft | 24h×8 | ×5 | 960 |
| 범용 RLVR/GRPO | 20_rlvr_grpo | 70h×8 | ×6 | 3,360 |
| 평가 | 40_eval | 10h×8 | ×8 | 640 |
| **합계** | | | | **4,960** |
</details>

### 데이터 (소스 확정 + 변환 검증)
- **Stage-1 (v3 혼합 콜드스타트)**: `neginb/OpenMedReason`(150,246·CC-BY-4.0·**게이트 auto**, 웹 동의 1회 필요) + `TIGER-Lab/VisualWebInstruct-verified`(97,295·MIT) + `UCSC-VLAA/VLAA-Thinking`(126,413·Apache-2.0). 전부 이미지 내장·다운로드 완료. 빌드 `build_mixed_coldstart.py` → [Stage-1](stage1_coldstart.md#stage-1--콜드스타트-sft).
  - ⚠️ VLAA 는 `vg`(38,242)·`coco`(8,727) = **46,969건의 이미지 tar 이 레포에 없음** → 해당 서브셋 제외(Visual Genome·COCO 별도 수급 시 복귀 가능). tar 보유: allava_laion·arxivqa·chartqa·clevr_math·docvqa·geoqa170k·synthesis·vizwiz(총 26.2GB).
- **Stage-2 (풀확장, 의료 27%)**: `skylenage-ai/DeepVision-103K`(일반, 서브샘플 40K) + `FanqingM/MMK12`(순수 math 15K) + `xmcmic/PMC-VQA`(**의료 MC** 20K, 329K 중 서브샘플). 조립 = `build_stage2_mix.py`(bytehash dedup) → train **74,787** / holdout 1,772. 탈락(실측): Kvasir(degenerate)·SLAKE(이미지450)·ThinkLite(노이즈)·PathVQA(절반개방형). 상세 → [`docs/stage2_data.md`](stage2_data.md). `DeepMath-103K`(텍스트) 혼동 주의.
- **Stage-3**: `MBZUAI/medix-rl-data`(51K, 개방형 의료 멀티모달). 둘 다 `work/hf_cache` 다운로드 완료·게이트 없음.
  - ⚠️ medix 는 **assistant 가 비어 있다**(prompt+solution 만) → 의료 추론 트레이스 없음. 정답도 `<MODALITY>\n자유서술`(중앙값 47자)이라 정확매칭 불가 = Stage-3 가 RaR judge 를 쓰는 이유.
- K-BDS 공개데이터 경로: `/kobic/ICECAP/DataStation/` · 매핑표 `/scratch/database/KBDSMAP/` · 업로드 `kbds-dm.kisti.re.kr`.
- 출력 포맷: `{"messages":[{"role":"user","content":"<image>...질문"}], "images":["/abs.png"], "solution":"정답"}`.

<details><summary>다운로드 → 변환 워크플로 (컨테이너 내, 로그인노드)</summary>

```bash
SB=work/images/ms-swift-413-sandbox
DV=$(ls -d work/hf_cache/hub/datasets--skylenage-ai--DeepVision-103K/snapshots/*/)
singularity exec --bind $PWD/work --env HF_HUB_OFFLINE=1 $SB python scripts/convert_to_swift.py \
  skylenage-ai/DeepVision-103K --parquet "${DV}math-77k.parquet" "${DV}visual_logic-26k.parquet" \
  --out work/data/deepvision103k_train.jsonl --images-dir work/data/images/deepvision
MX=$(ls -d work/hf_cache/hub/datasets--MBZUAI--medix-rl-data/snapshots/*/)
singularity exec --bind $PWD/work --env HF_HUB_OFFLINE=1 $SB python scripts/convert_to_swift.py \
  MBZUAI/medix-rl-data --parquet "${MX}data/train-*.parquet" \
  --out work/data/medix_rl_train.jsonl --images-dir work/data/images/medix
```
- ⚠️ parquet `List` feature가 datasets 구버전과 충돌 → 변환기는 **pyarrow 스트리밍**. 전체 변환은 이미지 ~154K개 추출(디스크·inode 큼).
</details>

### 홀드아웃 분리 (Stage-2 평가 누수 차단)
`scripts/make_holdout.py`: DeepVision엔 카테고리 라벨 없음 → **정답유형 프록시**(math=수치/수식, visual-logic=객관식 MC) 각 **1%** 층화 → `deepvision_holdout.jsonl`(972) / `deepvision103k_trainonly.jsonl`(102,531). 학습은 trainonly, 평가는 holdout. 평가 `eval_compare.py`(math/vl 층별 분리 보고).

#### ⚠️ 미해결: 홀드아웃 22% 중복 (2026-07-16 발견, 기록만 — 재측정 보류)

| 검사 | 결과 |
|---|---|
| 홀드아웃 972건 중 (질문+**이미지 경로**) 가 trainonly 와 동일 | **0건** ← 분리 로직 자체는 정상 |
| 홀드아웃 972건 중 (질문+**이미지 바이트해시**) 가 trainonly 와 동일 | **214건 (22.0%)** |
| ↳ 그중 **정답까지 동일**(= 완전중복) | **214건** (정답 불일치 0) |

**원인**: DeepVision 은 질문 중복이 많고(102,531행 → 고유 82,198, 중복 20,333) **같은 그림이 다른 경로에 중복 저장**돼 있다. `make_holdout.py` 는 경로 기준 dedup 이라 이를 못 걸렀다.

**영향**: 보고된 모든 홀드아웃 수치(base 0.15 / 콜드스타트 0.22 / dr_grpo 0.380 / GDPO 0.390 / GSPO 0.290 / ablation 0.18·0.165)는 **22% 만큼 상향편향**. 다만 clean 3종이 **동일한 22%**를 공유하므로 **상대비교·순위·판정 결론은 유효**(GDPO≈dr_grpo 동률, GSPO 열위). "clean vs 오염(DAPO·baseline 0.415)" 대비는 서술보다 약함 — clean 쪽도 22% 는 외운 문제.

**대응(합의)**: 지금은 **기록만** 하고 재측정 보류(GPU 예산 83% 소진). 향후 홀드아웃 재구성 시 **이미지 바이트해시 기준 dedup** 필수.

<details><summary>디렉토리 트리</summary>

```
kbds_project/
├── README.md · HANDOFF.md · plan.hwp
├── configs/
│   ├── accuracy.py          # Stage-2 보상 accuracy_mix + format_think
│   ├── medical_reward.py    # Stage-3 RaR 루브릭 보상 clinical_judge(AsyncORM)
│   └── ds_zero{2,3,3_offload}.json
├── scripts/                             # (총 49개 — 아래는 주요 스크립트)
│   ├── 00_common.sh                     # 공통 경로/환경/실행 래퍼
│   ├── 10_sft.slurm / 20_rlvr_grpo.slurm / 30_medical_rl.slurm / 40_eval.slurm  # 단계별 뼈대
│   │
│   │  # ── Stage-1 콜드스타트 v3 ──
│   ├── 11_build_coldstart.slurm         # 데이터 빌드 (cpu32 잡)
│   ├── build_mixed_coldstart.py         # 일반+의료 혼합, format_think==1.0 게이트
│   ├── 12_merge_mixed.slurm             # LoRA 병합 + 형식/길이 프로브
│   ├── 50_eval_v3.slurm / eval_v3_holdout.py   # 병합+홀드아웃 채점(strict format_think·층별·pred 덤프)
│   ├── 52_eval_v2_baseline.slurm        # v2 동일조건 재측정(A/B 확정용)
│   ├── plot_sft_curve.py                # SFT 학습곡선 → docs/assets/
│   │
│   │  # ── Stage-2 RLVR ──
│   ├── 21_rlvr_grpo_adv.slurm           # A/B(dapo/gspo/dr_grpo, RESUME/MAX_STEPS)
│   ├── make_holdout.py                  # 층화 홀드아웃 분리
│   ├── 40_eval_compare.slurm / eval_compare.py            # base vs 학습 벤치
│   ├── eval_midtrain.slurm              # 중간 홀드아웃 벤치(RL 25%)
│   ├── 46_eval_gdpo_ab.slurm / 47_eval_ablation.slurm     # GDPO 판정 · 콜드스타트 ablation
│   ├── 48_eval_gspo_holdout.slurm / 49_eval_contaminated.slurm  # GSPO 홀드아웃 · 오염 참고치
│   ├── plot_grpo_multi.py / plot_grpo_compare.py / plot_grpo_trend.py  # 기법 성능 plot
│   ├── merge_drgrpo.slurm / launch_stage3.sh / launch_gspo_ab.sh       # 오케스트레이터
│   │
│   │  # ── Stage-3 의료 RL ──
│   ├── judge_server.sh / 31_judge_smoke.slurm / 33_judge_probe.slurm   # judge 서빙·검증
│   ├── 34_rubric_compare.slurm / judge_compare_rubric.py  # 정적 vs 인스턴스 루브릭 비교
│   ├── 35_stage3_smoke.slurm            # 배선 end-to-end 스모크
│   ├── test_medical_reward.py           # 보상 유닛테스트(29)
│   │
│   │  # ── HealthBench / 공통 ──
│   ├── 45_healthbench_smoke.slurm / run_healthbench.py    # HealthBench Hard 하니스(n=1000 실측)
│   ├── _archive/                        # 폐기 스크립트(v1·v2 빌더 등, 이력 보존)
│   └── convert_to_swift.py / download_{model,dataset}.py
├── docs/ (medical_reward_spec.md · worklog_*.md) · logs/
```
</details>

### 사용 순서
```bash
# 0) 환경(1회, 로그인노드): bash env/build_image.sh  → work/images/ms-swift-413-sandbox
# 1) 경로/모델 확인: scripts/00_common.sh 상단 변수
# 2) 단계별 제출 (의존성 체이닝)
JID1=$(sbatch --parsable scripts/10_sft.slurm)
JID2=$(sbatch --parsable --dependency=afterok:$JID1 scripts/20_rlvr_grpo.slurm)
JID3=$(sbatch --parsable --dependency=afterok:$JID2 scripts/30_medical_rl.slurm)
sbatch          --dependency=afterok:$JID3 scripts/40_eval.slurm
```

---
