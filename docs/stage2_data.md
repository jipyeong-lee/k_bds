# Stage-2 데이터 (RLVR 풀확장) — 정리

> **갱신 2026-07-24.** Stage-2(범용 RLVR/GRPO)의 데이터 확장 전 과정: 동기·스크리닝·심층분석·혼합비율·빌드 파이프라인.
> 학습 세팅은 [`stage2_expansion_runbook.md`](stage2_expansion_runbook.md), 전체 현황은 [`../README.md`](../README.md).

---

## 1. 배경 & 동기

- **원래 Stage-2** = `DeepVision-103K` 단일 (trainonly 102,531 / 홀드아웃 972). GRPO A/B로 dr_grpo·GDPO 승자 확정, step600 홀드아웃 0.38~0.39 포화.
- **확장 동기**: v3 콜드스타트 A/B에서 **math가 v2와 동률(0.3245)** — 혼합 데이터가 vl(0.270→0.368)은 올렸으나 수치계산은 못 올림. 그리고 프로젝트 목표는 **의료**인데 Stage-2가 일반 전용이었음.
- **예산 해제**: 2026-07-22 다른 계정에 5,000 노드시간 확보 → Stage-2 재실행 여력 생김. 현재 계정은 세팅·검증 전담, GitHub 공유.
- **목표**: 검증가능 정답 RLVR로 **일반 시각추론 + math + 의료**를 균형 있게 강화.

## 2. 데이터셋 스크리닝 (전부 다운로드·실측 — 이름·규모로 안 고름)

| 소스 | 규모 | 고유이미지 | 고유답 | Yes/No | 검증가능 | 판정 |
|---|---|---|---|---|---|---|
| **DeepVision-103K** | 102,531 | 다수 | 8%(C/B/D 최빈) | — | 43% math_verify | ✅ 기존 base |
| **MMK12** | 15,616 | 다수 | 32% | ~0% | **82%**·100% math | ✅ **채택**(math 약점 직격) |
| **PMC-VQA** | 329,536 MC | **29,133/30K** | MC B/C/A/D 균형 | 낮음 | MC letter | ✅ **채택**(의료 광범위) |
| ThinkLite-VL-hard | 11,031 | 적음 | 35% | 11% | 49%·61% 취약 | ⚠️ 노이즈 → **드롭 권고** |
| ~~Kvasir-VQA~~ | 58,849 | **671**(17문항/img) | **107**·none 33% | 18% | — | ❌ **degenerate**(S-Chain급) |
| ~~SLAKE~~ | 4,919 | **450** | 220 | 34% | 클린 | ⏸ 스킵(이미지 450개뿐) |
| ~~VQA-RAD~~ | 1,793 | ~315 | — | 49% | — | ⏸ 극소 |
| ~~PathVQA~~ | 19,654 | — | — | 48% | 절반 개방형 | ⏸ 절반만 검증가능 |

**핵심 교훈**: "download and measure"가 반복 확인됨. Kvasir(58K인데 고유이미지 671·고유답 107), SLAKE(4.9K인데 이미지 450) 등 **규모와 다양성이 따로 논다**. 콜드스타트 때 S-Chain·MMFineReason과 같은 함정.

<details><summary>콜드스타트 때 이미 탈락시킨 것들 (RLVR 재검토)</summary>

MMK12·MM-Eureka·ThinkLite는 콜드스타트에서 "추론 trace 없음"으로 탈락했으나 **RLVR은 prompt+검증가능정답만 필요**해 오히려 적합 → MMK12·ThinkLite 재활용. MMFineReason(9~11K자 장황)은 여전히 보류.
</details>

## 3. 심층 분석 결과

**정답형식 분포 (검증가능성):**
- **DeepVision**: 49.5% MC letter + 27.8% int + 11.5% 수식 → 43% math_verify. **MC 과반·고유답 8%**라 정보량 낮고 step600 포화 원인.
- **MMK12**: int 46.5% + 수식 26.2% → **82% math_verify**, 100% 순수 math, 고유답 32%. 최고 품질 RLVR 소스.
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

## 6. 학습 연결 (Stage-2 GDPO)

- **init** = v3 `sft_mixed_merged`, **레시피** = GDPO (`--loss_type dr_grpo --scale_rewards gdpo`)
- ⚠️ 반드시 **`21_rlvr_grpo_adv.slurm`** 경유 — plateau 돌파 핵심(`dynamic_sample`+`overlong_filter`+`beta 0.04`)이 거기만 있음. `launch_stage2_expanded.sh`가 자동 라우팅.
- 보상: `accuracy_mix 1.0 + format_think 0.2 + soft_overlong 0.2`
- 평가: 확장 홀드아웃을 **소스별(`_source`) 분리 리포트** → 의료(PMC-VQA)에서 v3 대비 상승 + DeepVision 유지 확인.

---

**검증된 기준선**: v3 콜드스타트 홀드아웃 acc **0.348**(math 0.324=약점) / format_think **0.909**. v2+RL(구 DeepVision) 0.380~0.390. → 확장 Stage-2 성패 = 신규 홀드아웃(math·의료)에서 v3 대비 상승.
