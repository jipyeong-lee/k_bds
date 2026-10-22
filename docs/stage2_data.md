# Stage-2 데이터 (RLVR 풀확장) — 정리

> **갱신 2026-07-24.** Stage-2(범용 RLVR/GRPO)의 데이터 확장 전 과정: 동기·스크리닝·심층분석·혼합비율·빌드 파이프라인.
> 재현(빌드) 절차는 아래 [§5-1 빌드 런북](#5-1-빌드-런북), 전체 현황은 [`../README.md`](../README.md).

---

## 1. 배경 & 동기

- **원래 Stage-2** = `DeepVision-103K` 단일 (trainonly 102,531 / 홀드아웃 972). GRPO A/B로 dr_grpo·GDPO 승자 확정, step600 홀드아웃 0.38~0.39 포화.
- **확장 동기**: v3 콜드스타트 A/B에서 **math가 v2와 동률(0.3245)** — 혼합 데이터가 vl(0.270→0.368)은 올렸으나 수치계산은 못 올림. 그리고 프로젝트 목표는 **의료**인데 Stage-2가 일반 전용이었음.
- **예산 해제**: 2026-07-22 다른 계정에 5,000 노드시간 확보 → Stage-2 재실행 여력 생김. 현재 계정은 세팅·검증 전담, GitHub 공유.
- **목표**: 검증가능 정답 RLVR로 **일반 시각추론 + math + 의료**를 균형 있게 강화.

## 2. 데이터셋 스크리닝 (전부 다운로드·실측 — 이름·규모로 안 고름)

| 소스 | 설명 | 규모 | 고유이미지 | 고유답 | Yes/No | 검증가능 | 판정 |
|---|---|---|---|---|---|---|---|
| **DeepVision-103K** | 수학·시각논리 멀티모달(일반) | 102,531 | 다수 | 8%(C/B/D 최빈) | — | 43% math_verify | ✅ 기존 base |
| **MMK12** | 중국 K-12 수학 시험문제(영어 번역본, 이미지) | 15,616 | 다수 | 32% | ~0% | **82%**·100% math | ✅ **채택**(math 약점 직격) |
| **PMC-VQA** | PubMed 논문 그림 의료 VQA(4지선다) | 329,536 MC | **29,133/30K** | MC B/C/A/D 균형 | 낮음 | MC letter | ✅ **채택**(의료 광범위) |
| ThinkLite-VL-hard | 하드 멀티모달 추론(차트·표·도형 단답) | 11,031 | 적음 | 35% | 11% | 49%·61% 취약 | ⚠️ 노이즈 → **드롭** |
| ~~Kvasir-VQA~~ | 위장관 내시경 이미지 VQA | 58,849 | **671**(17문항/img) | **107**·none 33% | 18% | — | ❌ **degenerate**(S-Chain급) |
| ~~SLAKE~~ | 방사선(CT/MRI/X-ray) 의료 VQA(영어판) | 4,919 | **450** | 220 | 34% | 클린 | ⏸ 스킵(이미지 450개뿐) |
| ~~VQA-RAD~~ | 방사선 VQA(임상의 작성) | 1,793 | ~315 | — | 49% | — | ⏸ 극소 |
| ~~PathVQA~~ | 병리 조직 슬라이드 VQA | 19,654 | — | — | 48% | 절반 개방형 | ⏸ 절반만 검증가능 |

**핵심 교훈**: "download and measure"가 반복 확인됨. Kvasir(58K인데 고유이미지 671·고유답 107), SLAKE(4.9K인데 이미지 450) 등 **규모와 다양성이 따로 논다**. 콜드스타트 때 S-Chain·MMFineReason과 같은 함정.

<details><summary>콜드스타트 때 이미 탈락시킨 것들 (RLVR 재검토)</summary>

MMK12·MM-Eureka·ThinkLite는 콜드스타트에서 "추론 trace 없음"으로 탈락했으나 **RLVR은 prompt+검증가능정답만 필요**해 오히려 적합 → MMK12·ThinkLite 재활용. MMFineReason(9~11K자 장황)은 여전히 보류.
</details>

## 3. 심층 분석 결과

**정답형식 분포 (검증가능성):**
- **DeepVision**: 49.5% MC letter + 27.8% int + 11.5% 수식 → 43% math_verify. **MC 과반·고유답 8%**라 정보량 낮고 step600 포화 원인.
- **MMK12**: int 46.5% + 수식 26.2% → **82% math_verify**, 100% 순수 math, 고유답 32%. 최고 품질 RLVR 소스. **언어=영어 번역본**(실측: 15,616건 중 한자 잔존 52건·0.3%, 대부분 `甲乙`(물체 A/B)·`保留`(반올림 지시) 자투리).
- **PMC-VQA**: MC 4지선다, Answer_label 균형(B 54K/C 51K/A 43K/D 28K). OpenMedReason의 A편중(77%)과 대조.
- **ThinkLite**: 34% 수치 + **61% 짧은문자열**(times/장소/Yes-No 11%). 다양성은 있으나 노이즈.

**reward 견고성 실측 (중요):** `accuracy_mix`가 "맞지만 형식 다른" 답을 0점 처리하는지 12케이스 테스트 → **0/12 손실**. 콤마(`2,380`↔`2380`)·분수(`3/4`↔`0.75`)·단위·시간·대소문자·Yes/No 전부 1.0. **math_verify + casefold가 이미 견고** → 문자열 정규화 불필요.

**의료 VQA 공통 함정**: 대부분 40~50% Yes/No(이진 50% 추측)·저다양성. PMC-VQA만 대형·MC·광범위로 예외.

## 4. 혼합비율 결정

**문제**: GRPO는 셔플 샘플링이라 자연비율(DeepVision 80%)이면 롤아웃 80%가 이미 포화된 데이터 → **확장 의미 반감**. 600~1000 step × 배치 32 = ~2~3만 prompt만 봄(DeepVision 102K는 다 못 봄) → **비율만 중요, DeepVision 다 넣을 이유 없음**.

**권장 구성 (의료 27%)** — 2026-07-24:

| 소스 | 학습 | 비중 | 역할 |
|---|---|---|---|
| DeepVision (서브샘플) | 40,000 | 53% | 일반 시각추론 base (RL 검증됨) |
| MMK12 (전량) | 15,207 | 20% | 순수 math (유일한 측정 약점 직격) |
| **PMC-VQA** (서브샘플) | 20,000 | 27% | **의료 광범위 MC** (프로젝트 목표 정렬) |
| **합계** | **~75K** | | 일반 53 / math 20 / **의료 27** |

- **왜 의료 40%까지 안 가나**: PMC-VQA는 "그림 인식+MC"라 가장 깊은 추론은 아님. **깊은 의료는 Stage-3(medix+RaR judge)가 전담** → Stage-2를 의료-MC로 과적하면 전이 추론보다 패턴매칭 학습 위험. (의료 최우선이면 PMC 28K→40% 대안)
- **ThinkLite 드롭**: 가장 노이즈 큰 소스, 일반 단답 역할은 DeepVision과 중복.
- **오염 방지**: 신규 홀드아웃은 **이미지 바이트해시 dedup**(구 DeepVision 홀드아웃 22% 오염 재발 방지).

### Stage-2 ↔ Stage-3 관계 (cascade + replay)
- Stage-2(검증가능, accuracy_mix) → Stage-3(개방형 medix, RaR judge), init=Stage-2 ckpt. **RLVR로 추론 습관 다지고 → judge로 정교화**하는 curriculum.
- **의료 VQA(Stage-2 검증가능) ↔ medix(Stage-3 개방형)는 상보적.** 단 같은 출처(PMC-VQA/SLAKE는 medix 출처)라 **홀드아웃은 dedup 필수**(학습 중복은 무해).
- 통합(단일 스테이지 verifiable+judge 혼합) 대비 cascade가 curriculum·judge인프라 단순·디버깅에서 우위. 망각은 Stage-3에 검증가능 20~30% replay로 완화.

## 5. 현재 상태 & 빌드 파이프라인

**✅ 재조립 완료 (2026-07-24, `build_stage2_mix.py`):**

| | 소스 | 건수 | 비중 |
|---|---|---|---|
| **train** | DeepVision(서브샘플·오염453 제외) | 40,000 | 53% |
| | MMK12(전량) | 15,204 | 20% |
| | PMC-VQA(서브샘플 20K) | 19,583 | 26% |
| | **합계** | **74,787** | 일반53/math20/의료26 |
| **holdout** | DeepVision 972 + MMK12 400 + PMC-VQA 400 | **1,772** | `_source` 태그 |

- **전 소스 bytehash dedup 검증: 누수 0** (deepvision·mmk12·pmcvqa 홀드아웃 전부 train과 이미지해시 중복 0).
- **구 DeepVision 22% 오염 최종 해소**: 홀드아웃 972의 이미지해시를 train 서브샘플에서 제외(453건). 이제 확장 홀드아웃 전체가 클린.
- 산출: `work/data/stage2_expanded_{train,holdout}.jsonl`.

**스크립트:**
- `scripts/convert_to_swift.py` — parquet→swift (DeepVision/medix/MMK12/ThinkLite/SLAKE 자동감지)
- `scripts/build_pmcvqa.py` — PMC-VQA CSV+MC+zip 선택추출 (letter 균형, seed42)
- `scripts/13_build_stage2_expanded.slurm` — 신규 소스 변환 CPU 잡 (conda swift env, 컨테이너 불필요)
- `scripts/build_stage2_mix.py` — 확장셋 조립 + 소스별 층화 홀드아웃 (bytehash dedup, `DV_CAP`/`PMC_CAP` env)

> **비율 조정**: `DV_CAP=33000 PMC_CAP=28000 python3 scripts/build_stage2_mix.py` 로 의료 40% 등 재조립 가능.

**남은 소스(미사용, 재현용 보존)**: `thinklite_train.jsonl`(11,031·노이즈로 드롭), `pmcvqa_train.jsonl`(30K 풀 중 20K 사용), SLAKE(변환 안 함).

### 5-1. 빌드 런북

*(구 `stage2_expansion_runbook.md` 에서 옮김. 그 문서의 학습 단계(구 혼합 체인 `launch_stage2_expanded*.sh`)는 붕괴로 폐기된 경로라 맨 아래에 기록만 남긴다. 현재 학습 진입점은 [`README.md`](../README.md) "Stage-2 도메인 전문가 학습".)*

> 대용량 데이터(이미지·jsonl·체크포인트)는 git 에 없으므로 아래로 **재생성**한다.
> **같은 클러스터라면 재생성보다 직접전송이 훨씬 빠름** → [`../HANDOFF.md`](../HANDOFF.md) §2-A.
>
> 🚨 **2026-07-27~ 클러스터 apptainer 파손**으로 `singularity` 실행·빌드 모두 불가. 아래 컨테이너 명령은
> **그대로는 동작하지 않는다.** 학습·추론은 `ENV_MODE=loader` 우회(기본값)로 수행되며, 데이터 다운로드/변환은
> conda 경로를 쓴다. → [`../HANDOFF.md`](../HANDOFF.md) §3 · `../runc.sh`

**0) 사전 (1회)**
- 레포 clone, `scripts/00_common.sh` 상단 경로 확인(`PROJ_DIR`/`WORK_DIR`).
- 컨테이너 이미지: `work/images/ms-swift-413-sandbox` (직접전송 권장). ⚠️ `bash env/build_image.sh` 는 apptainer 파손으로 **현재 불가**.
- HF 토큰(게이트/rate-limit): `export HF_TOKEN=hf_...`

**1) 데이터 다운로드 (로그인노드)** — 아래는 apptainer 정상 시의 명령. 현재는 `singularity exec ...` 대신 pyarrow/requests 있는 conda python 으로 `scripts/download_dataset.py` 를 직접 실행할 것.
```bash
SB=work/images/ms-swift-413-sandbox
for ds in skylenage-ai/DeepVision-103K MBZUAI/medix-rl-data FanqingM/MMK12 xmcmic/PMC-VQA; do
  env -u HF_HUB_OFFLINE singularity exec --bind "$PWD/work:$PWD/work" \
    --env HF_HUB_OFFLINE=0 --env HF_HOME="$PWD/work/hf_cache" --env HF_TOKEN="$HF_TOKEN" \
    "$SB" python scripts/download_dataset.py "$ds"
done
```
⚠️ 무인증이면 rate-limit 로 큰 parquet 이 멈출 수 있음 → `HF_TOKEN` 필수. (겪은 이슈: MMK12 train-00000 정체 → 토큰으로 해결.)

**2) 변환 → swift jsonl (CPU 잡, 컨테이너 불필요)** — DeepVision·medix 는 컨테이너 변환 워크플로가 [`ops_data.md`](ops_data.md) 에 있다. 신규 2종은 CPU 잡으로:
```bash
sbatch scripts/13_build_stage2_expanded.slurm     # conda swift env(pyarrow+PIL) 로 MMK12 변환
#   ⚠️ 계정 이식 시 BUILD_PY 지정 필요: export BUILD_PY=<pyarrow+PIL 있는 python>
# 산출: work/data/mmk12_train.jsonl + work/data/images/mmk12
```
> 계산노드는 컨테이너 불가(userns=0)이나 변환은 pyarrow+PIL 만 필요 → conda `swift` env 로 CPU 노드서 실행.

**3) 확장셋 조립 + 클린 홀드아웃**
```bash
# 신규 소스 변환: MMK12(parquet) 는 13_build 로, PMC-VQA(CSV+zip) 는 build_pmcvqa 로
python3 scripts/build_pmcvqa.py --n 30000 --csv <train.csv> <train_2.csv> \
  --zips <images.zip> <images_2.zip> --out work/data/pmcvqa_train.jsonl --images-dir work/data/images/pmcvqa
python3 scripts/build_stage2_mix.py     # seed=42, bytehash dedup, 의료 27% 기본(실제 26%)
# train   = DeepVision 40,000 + MMK12 15,204 + PMC-VQA 19,583 = 74,787 (일반53/math20/의료26)
# holdout = DeepVision 972 + MMK12 400 + PMC-VQA 400 = 1,772  (_source·_stratum 태그)
python3 scripts/split_stage2_by_source.py          # 전문가용 소스별 3분할 (--dry / --verify)
```

**4) v3 콜드스타트 init 준비**
```bash
sbatch scripts/10_sft.slurm                       # v3 SFT → sft_mixed_lora/checkpoint-298
sbatch scripts/50_eval_v3.slurm                   # 병합(sft_mixed_merged) + 홀드아웃 평가(선택)
#  (병합만 필요하면 12_merge_mixed.slurm)
```
※ v3 SFT 데이터(`sft_mixed_*`)도 재현하려면 `scripts/{11_build_coldstart.slurm, build_mixed_coldstart.py}`.

**5) (폐기 경로, 기록용) 구 혼합 GRPO 실행**
```bash
SMOKE=1 bash scripts/launch_stage2_expanded.sh    # 배선 확인(max_steps 5)
bash scripts/launch_stage2_expanded_epoch.sh      # 2,337 step(=0.25 epoch) 체인 4잡 ≈209h
#   단발(≤70h): MAX_STEPS=600 bash scripts/launch_stage2_expanded.sh · RECIPE=dr_grpo 선택 가능 · 체인 중단 시 4개 job 전부 scancel
```
- 런처는 `21_rlvr_grpo_adv.slurm` 를 제출(20 아님) — `dynamic_sample`+`overlong_filter`+`beta 0.04` 는 21 에만 있었다. 산출 `work/checkpoints/grpo_expanded_gdpo`. 이 경로는 step ~900 에서 붕괴했다 → [사후분석](stage2_run73924_postmortem.md).

**6) 평가 (확장 홀드아웃)** — 병합 후 vLLM serve → `stage2_expanded_holdout.jsonl` 채점(`EVAL_DATA=work/data/stage2_expanded_holdout.jsonl`, `50_eval_v3.slurm`·`eval_midtrain.slurm` 패턴). `_source`·`_stratum` 별 분리 리포트.

## 6. 학습 연결 (Stage-2 GDPO)

- **init** = v3 `sft_mixed_merged`, **레시피** = GDPO (`--loss_type dr_grpo --scale_rewards gdpo`)
- ⚠️ 반드시 **`21_rlvr_grpo_adv.slurm`** 경유 — plateau 돌파 핵심(`dynamic_sample`+`overlong_filter`+`beta 0.04`)이 거기만 있음. `launch_stage2_expanded.sh`가 자동 라우팅.
- 보상: `accuracy_mix 1.0 + format_think 0.2 + soft_overlong 0.2`
- 평가: 확장 홀드아웃을 **소스별(`_source`) 분리 리포트** → 의료(PMC-VQA)에서 v3 대비 상승 + DeepVision 유지 확인.

---

**검증된 기준선**: v3 콜드스타트 홀드아웃 acc **0.348**(math 0.324=약점) / format_think **0.909**. v2+RL(구 DeepVision) 0.380~0.390. → 확장 Stage-2 성패 = 신규 홀드아웃(math·의료)에서 v3 대비 상승.
