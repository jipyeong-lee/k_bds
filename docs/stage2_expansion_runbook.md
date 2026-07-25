# Stage-2 풀확장 — 재현 런북 (다른 계정용)

**목적**: Stage-2 RLVR 을 **DeepVision 단일 → DeepVision + MMK12 + ThinkLite-VL 확장**으로 재실행.
v3 콜드스타트(`sft_mixed_merged`)를 init 으로, 확장 RLVR 데이터로 GRPO. v3 약점(math 동률)을 STEM 다양성으로 겨냥.

> 이 계정(k252a01)에서 **세팅·검증은 완료**했고 GitHub 에 전부 커밋돼 있음. 대용량 데이터(이미지·jsonl·체크포인트)는 git 에 없으므로 아래로 **재생성**한다. 실제 RL 본실행(~500 노드시간)은 예산 있는 계정에서 수행.

## 0. 사전 (1회)
- 레포 clone, `scripts/00_common.sh` 상단 경로 확인(`PROJ_DIR`/`WORK_DIR`).
- 컨테이너 이미지 빌드: `bash env/build_image.sh` → `work/images/ms-swift-413-sandbox`.
- HF 토큰(게이트/rate-limit): `export HF_TOKEN=hf_...`

## 1. 데이터 다운로드 (로그인노드, 컨테이너)
```bash
SB=work/images/ms-swift-413-sandbox
for ds in skylenage-ai/DeepVision-103K MBZUAI/medix-rl-data FanqingM/MMK12 russwang/ThinkLite-VL-hard-11k; do
  env -u HF_HUB_OFFLINE singularity exec --bind "$PWD/work:$PWD/work" \
    --env HF_HUB_OFFLINE=0 --env HF_HOME="$PWD/work/hf_cache" --env HF_TOKEN="$HF_TOKEN" \
    "$SB" python scripts/download_dataset.py "$ds"
done
```
⚠️ 무인증이면 rate-limit 로 큰 parquet 이 멈출 수 있음 → `HF_TOKEN` 필수. (겪은 이슈: MMK12 train-00000 정체 → 토큰으로 해결.)

## 2. 변환 → swift jsonl (CPU 잡, 컨테이너 불필요)
DeepVision·medix 는 컨테이너로 이미 변환하는 워크플로가 README 에 있음. 신규 2종은 CPU 잡으로:
```bash
sbatch scripts/13_build_stage2_expanded.slurm     # conda swift env(pyarrow+PIL) 로 MMK12·ThinkLite 변환
# 산출: work/data/{mmk12,thinklite}_train.jsonl + work/data/images/{mmk12,thinklite}
```
> 계산노드는 컨테이너 불가(userns=0)이나 변환은 pyarrow+PIL 만 필요 → conda `swift` env 로 CPU 노드서 실행.

## 3. 확장셋 조립 + 클린 홀드아웃
```bash
# 신규 소스 변환: MMK12·ThinkLite (parquet) 는 13_build 로, PMC-VQA (CSV+zip) 는 build_pmcvqa 로
python3 scripts/build_pmcvqa.py --n 30000 --csv <train.csv> <train_2.csv> \
  --zips <images.zip> <images_2.zip> --out work/data/pmcvqa_train.jsonl --images-dir work/data/images/pmcvqa
python3 scripts/build_stage2_mix.py     # seed=42, bytehash dedup, 의료 27% 기본
# train   = DeepVision 40,000 + MMK12 15,204 + PMC-VQA 19,583 = 74,787 (일반53/math20/의료26)
# holdout = DeepVision 972 + MMK12 400 + PMC-VQA 400 = 1,772  (_source·_stratum 태그)
```
- **전 소스 이미지 바이트해시 dedup**(구 DeepVision 22% 오염 최종 해소). 검증됨: 전 소스 홀드아웃 1,772건 train 누수 0.
- 비율 조정: `DV_CAP=33000 PMC_CAP=28000 python3 scripts/build_stage2_mix.py`(의료 40%).
- 데이터 상세·근거: [`stage2_data.md`](stage2_data.md).

## 4. v3 콜드스타트 init 준비
```bash
sbatch scripts/10_sft.slurm                       # v3 SFT → sft_mixed_lora/checkpoint-298
sbatch scripts/50_eval_v3.slurm                   # 병합(sft_mixed_merged) + 홀드아웃 평가(선택)
#  (병합만 필요하면 12_merge_mixed.slurm)
```
※ v3 SFT 데이터(`sft_mixed_*`)도 재현하려면 `scripts/{11_build_coldstart.slurm, build_mixed_coldstart.py}`.

## 5. Stage-2 확장 GRPO 실행
```bash
SMOKE=1 bash scripts/launch_stage2_expanded.sh    # 배선 확인(max_steps 5) — 먼저 권장
bash scripts/launch_stage2_expanded.sh            # 본실행 (기본=GDPO, ~70h/8gpu)
# RECIPE=dr_grpo 로 dr_grpo 선택 가능
```
- **런처는 검증된 `21_rlvr_grpo_adv.slurm` 를 제출**(20 아님) — GDPO/dr_grpo 의 plateau 돌파 핵심 `dynamic_sample`+`overlong_filter`+`beta 0.04` 는 21 에만 있음.
- **GDPO** = `--loss_type dr_grpo --scale_rewards gdpo` (보상별 advantage 개별정규화). A/B 검증(job 59191, 홀드아웃 0.390).
- 보상: `accuracy_mix 1.0 + format_think 0.2 + soft_overlong 0.2` (configs/accuracy.py).
- **하이퍼파라미터 외부 관행 대조·A/B knob** → [`rlvr_hparams_external.md`](rlvr_hparams_external.md). 기본값은 검증된 값(그룹 4·β 0.04·temp 0.9) 유지, `NUM_GEN=8`/`TEMPERATURE=1.0`/`BETA=0.01` 로 override 실험. **에포크는 늘리지 말 것**(≤1ep, 50스텝마다 홀드아웃, 포화 시 조기중단 — 다양성 붕괴 방지).
- 산출: `work/checkpoints/grpo_expanded_gdpo`.

## 6. 평가 (확장 홀드아웃)
```bash
# 병합 후 vLLM serve → eval_v3_holdout.py 로 stage2_expanded_holdout.jsonl 채점
EVAL_DATA=work/data/stage2_expanded_holdout.jsonl ... (50_eval_v3.slurm 패턴 재사용)
```
- `_source`(deepvision/mmk12/thinklite)·`_stratum` 별 분리 리포트 → 확장 이득이 **신규 STEM 분포**에서 나오는지 확인.

## 참고 — 검증된 기준선 (이 계정 실측)
| | format_think | holdout acc |
|---|---|---|
| v2 콜드스타트(동일하니스) | 0.185 | 0.295 |
| **v3 콜드스타트** | **0.909** | **0.348** (math 0.324 = v3 약점) |
| v2+RL dr_grpo/GDPO(step600, 구 DeepVision) | — | 0.380 / 0.390 |

→ 확장 Stage-2 의 성패 판단: **신규 홀드아웃(MMK12/ThinkLite)에서 v3(0.348) 대비 상승** + DeepVision 유지.
