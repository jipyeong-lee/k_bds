# 진행 이력 · 의사결정 · TODO

> 이 문서는 [`README.md`](../README.md) 에서 분리된 상세 기록입니다. 요약·현황은 README, 상세는 여기.

## 진행 이력 & 체크리스트

### 스테이지별 진행 사항

**⚠️ 계획 순서대로 진행되지 않았다.** ②를 돌리다 막힌 원인이 사실 ①이었음을 밝혀내고 되돌아온 것이 이 프로젝트의 핵심 서사다.

```
계획:  ① 콜드스타트 → ② RLVR → ③ 의료RL → ④ 평가
실제:  ② 먼저(06-16~07-13) → ③ 배선 선행(06-28~07-01) → ① 로 회귀(07-08~07-20) → ④
```

| 스테이지 | 기간 | 상태 | 핵심 결과 |
|---|---|---|---|
| **기반** | 06-15 | ✅ | NVLink 부재 발견 → **전 단계 LoRA** 강제 (full-FT 375~660s/step) |
| **① 콜드스타트 SFT** | 06-15~16, 07-08~20 | ✅ **v3 평가완료** | 세 번 재설계(v1→v2→v3). **형식 천장 0.473 → 0.909 완파**, acc 0.348 |
| **② 범용 RLVR** | 06-16~07-13 · 확장 07-22~24 · 본실행 07-28~ | ✅ 방법론 종결 · 🚀 **본실행 중** | dr_grpo 승자(Acc 0.526). **풀확장**(DeepVision 40K+MMK12+PMC-VQA 의료=74,787, init=v3·GDPO) → 2,337 step(=0.25 epoch) 체인 job **73924~73927** · step 630 시점 [중간점검](stage2_run73924_progress.md) |
| **③ 의료 RL** | 06-28~07-01 | 🟡 배선완료·본실행 대기 | RaR 루브릭·judge(27B) 검증·**e2e 스모크 PASS** |
| **④ 평가** | 전 기간 산발 | 🔄 진행 | 누수·오염 발견마다 재측정. v3 홀드아웃 0.348 |

#### ① 콜드스타트 SFT — 세 번 다시 만든 단계

| 버전 | 시점 | **왜 만들었나** | 어떻게 | 결과 |
|---|---|---|---|---|
| **v1** | 06-15 | ② 착수를 위한 형식 주입이 급했음 | VLAA `clevr_math` 단일 2,913 | ❌ 도메인 단일 → 일반화 실패, 폐기 |
| **v2** | 06-16 | base 장문추론(3.5~4.6K토큰) → 잘림 → **형식0**. ZeRO-3 길이확대는 5배 느려 불채택 → **추론을 짧게** 가르쳐야 했음 | rejection sampling(STaR/ReST) 자기증류 **727** | ✅ Stage-2 성공 발판. 그러나 **RL 형식 0.425 정체** |
| *검증* | 07-08~09 | "이 단계가 순가치를 하나?" | **2×2 ablation** — 콜드스타트 없이 base→RL | 없으면 홀드아웃 **0.18/0.165**(SFT 단독 0.22에도 미달) → **필수 확정** |
| **v3** | 07-16~20 | ⭐ **0.425 정체의 진짜 원인 발견** — `format_think` 를 v2 *데이터* 에 돌리니 **0.473**. RL 실패가 아니라 **데이터가 천장**이었음 | 게이트 `==1.0` 강제 + **의료 CoT 최초 투입** + 전수 실측 스크리닝 | 🔥 **format_think 0.909 · acc 0.348** |

> **v2 결함의 근원**: `build_rft_coldstart.py` 게이트가 느슨(`</think>`·`<answer>` 존재만, 앵커링·최소길이 없음) + **"가장 짧은 3개"** 선별이 객관식 **찍기**(`<think></think><answer>C</answer>`)를 우선 채택.

#### ② 범용 RLVR — 막힐 때마다 진단→처방

| 시점 | **막힌 것 / 왜** | 처방 | 결과 |
|---|---|---|---|
| 06-16 | 추론 길이 폭주 → 잘림 → 형식0 | ZeRO-3 길이확대 | ❌ 5배 느림 → 불채택 (→ ① v2 로 우회) |
| 06-19 | **Acc plateau** — `zero_std` 0.24→0.33, 그룹 내 보상 동일 → advantage 0 → 그래디언트 소멸 | **DAPO**(동적 샘플링으로 zero-std 제거) | 안정성↑이나 **미돌파 → 종결** |
| 06-25~28 | ″ 정규화 편향 의심 | **dr_grpo**(두 정규화 편향 제거) | ✅ **돌파** Acc 0.526 → **승자** `checkpoint-600` |
| 07-01~04 | 최신기법 미검증 | **GSPO** A/B | 동률. 홀드아웃 **0.290** = 일반화 실패 → 미채택 |
| 07-04~07 | ″ | **GDPO** A/B | 동률·downside 없음 → **Stage-3용 채택 권고** |
| 07-01 | **평가 누수** — 평가셋이 학습파일 stride 슬라이스 | **층화 홀드아웃 972 분리** + fresh 재학습 | "+67%" 무효 폐기, 정식 수치로 대체 |

#### ③ 의료 RL — 배선만 선행

| 시점 | 작업 | **왜 이 시점에** | 결과 |
|---|---|---|---|
| 06-28 | RaR 루브릭 `medical_reward.py` | ② 승자 확정 직후 착수 | 유닛테스트 통과 |
| 06-29 | judge(Qwen3.6-27B-FP8) 검증 | **judge 서버가 비자명한 인프라**라 조기 검증 필요 | 단일40GB·멀티모달·단조성 OK |
| 06-29 | 내부망 도달성 · 분포 프로브 | 컴퓨트노드 오프라인 제약 | 4차원 보상 전부 활성 |
| 06-29 | 정적 vs 인스턴스 루브릭 비교 | 설계 선택 | **정적 채택**(시각근거 변별 우세) |
| 07-01 | **e2e 스모크** | 본실행 전 검증 | ✅ PASS (`images` dict 실버그 수정, 유닛 29/29) |

#### ④ 평가 — 누수·오염 발견마다 재측정

| 시점 | **계기(왜 다시 쟀나)** | 결과 |
|---|---|---|
| 07-01 | stride 슬라이스 **누수** 발견 | 층화 홀드아웃 재구축, "+67%" 폐기 |
| 07-03 | 전량 학습 계속할지 판단 필요 | init 0.22 → trained **0.38**(+73%) → 계속 확정 |
| 07-07 | 의료 기준선 필요 | HealthBench base **0.229** / 콜드스타트 **0.224** |
| 07-07 | 기법 최종 판정 | 홀드아웃 GDPO **0.390** ≈ dr_grpo **0.380** |
| 07-13 | GSPO만 홀드아웃 미측정 | **0.290** = 일반화 실패 확인 |
| 07-16 | — | 홀드아웃 **22% 오염** 발견(기록만, 재측정 보류) |
| 07-20 | v3 검증 | **0.348 / format_think 0.909**. v3 는 DeepVision 미학습이라 **오염 이득 없음** → 비교가 v3 에 보수적 |

### 핵심 의사결정 (요약)
- **NVLink 없음 → 전 단계 LoRA** (full-FT 375~660s/step → LoRA ~5배↑).
- **추론 길이 폭주 → 간결 콜드스타트** (ZeRO-3 길이확대는 5배 느려 불채택). **Ablation으로 필수성 입증**(없으면 base→RL 형식0 붕괴·홀드아웃 0.18).
- **형식보상 천장 발견 → 콜드스타트 v3 재설계**(2026-07-16). v2 데이터 자체가 `format_think` **0.473**(느슨한 `closed()` 필터 + "가장 짧은 것" 선별이 찍기를 우선) → RL 형식 0.425 정체의 진짜 원인. **게이트를 1.0 으로 강제** + 프로젝트 최초로 **의료 CoT 투입**(medix 는 추론 트레이스가 0건).
- **데이터는 반드시 받아서 실측 후 채택**(2026-07-16). 이름·초록으로 고르면 틀린다 — S-Chain("Structured Visual CoT for Medicine")은 실제로 **고유정답 3개**, MMFineReason 은 trace 중앙값 **9~11K자**, MMK12/MM-Eureka 는 **추론 trace 자체가 없음**. 전부 다운로드·파싱 후에야 드러남.
- **plateau 돌파 → dr_grpo** (두 정규화 편향 제거로 zero-std plateau 통과). GSPO·GDPO도 A/B로 검증 → 동률(GDPO만 Stage-3용 채택).
- **평가 누수 차단 → 층화 홀드아웃 + fresh 1 epoch 재학습** (기존 stride 슬라이스는 학습 파일과 겹침).
- **Stage-3 → 정적 RaR 루브릭** (인스턴스식 대비 시각근거 변별 우세).

### 날짜별 이력

<details><summary>일별 상세 worklog (클릭)</summary>

- [2026-06-15](worklog_2026-06-15.md) — 초기 세팅·데이터 수급
- [2026-06-16](worklog_2026-06-16.md) — 추론모드·길이 전략 정립
- [2026-06-17](worklog_2026-06-17.md) — Stage-2 추세 확인 & 운영 정리
- [2026-06-19](worklog_2026-06-19.md) — Stage-2 baseline 완주 & plateau 진단
- [2026-06-22](worklog_2026-06-22.md) — GRPO 파생기법 A/B 실험 착수
- [2026-06-24](worklog_2026-06-24.md)
- [2026-06-25](worklog_2026-06-25.md)
- [2026-06-28](worklog_2026-06-28.md)
- [2026-06-29](worklog_2026-06-29.md)
- [2026-07-01](worklog_2026-07-01.md)

</details>

- **06-15~17** — 환경·모델·데이터 확정. NVLink 부재 발견→LoRA 전환. format 콜드스타트, `accuracy_mix`, GRPO 파일럿. Stage-2 착수(57249).
- **06-19** — Stage-2 baseline 완주(step1000). **Acc plateau 진단**(zero_std 0.24→0.33).
- **06-22~24** — plateau 돌파 A/B config(`21_..adv`). **DAPO 착수**(57527) → 안정성↑(grad 무spike·zero_std 0.00)이나 Acc 적신호. step600 돌파판정 자동화.
- **06-25** — **DAPO 종결**(미돌파). **dr_grpo 착수**(57624).
- **06-28** — **dr_grpo 돌파 확인**(step501~600 Acc 0.526>0.500). 승자 = `checkpoint-600`.
- **06-29** — **Stage-3 착수**: RaR 루브릭·`medical_reward.py`·judge(Qwen3.6-27B-FP8) 확정. **정적 vs 인스턴스 비교→정적 채택**.
- **06-30** — **홀드아웃 정비 + 1 epoch 재학습 착수**(층화 972, trainonly 102,531, fresh, MAX_STEPS 3204). +67% 무효화. 속도 병목 진단(~365s/step 구조적).
- **07-01** — **Stage-3 배선 end-to-end 스모크**: images dict 실버그 발견·수정, 유닛 29/29, 재검증 PASS. GRPO/DAPO/dr_grpo/GSPO 논문 정리. **GSPO A/B 착수**(59004, dr_grpo 병렬).
- **07-02** — 병렬 진행: dr_grpo step~656/3204, GSPO step~198/600. dr_grpo 판정창 501~600 완성(Acc 0.487, on-policy). **예비 비교(동일 step 100~200)**: GSPO ≈ dr_grpo **동률**(Acc 0.516 vs 0.510), GSPO 클리핑↑ → 아직 우위 신호 없음. README 전면 재구조화(427→331줄).
- **07-03** — **중간 홀드아웃 벤치마크**(`eval_midtrain.slurm`, RL 25%=step 800, 층화 N=100): **init 0.22 → trained 0.38(+73%)**, base 0.15 대비 +153% → **RL 홀드아웃 개선 확인, 전량 학습 계속 확정**(첫 유효 홀드아웃 수치, 무효 +67% 대체). dr_grpo step~883, GSPO step~466.
- **07-04** — **GSPO A/B 완료·미채택**(판정창 동률 → dr_grpo 유지). **GDPO A/B 착수**(`job 59191`, `RECIPE=dr_grpo SCALE_REWARDS=gdpo` = dr_grpo 처리 유지 + advantage만 보상별 개별정규화). RLVR 방법론 종합비교(GRPO/DAPO/dr_grpo/GSPO/GDPO) README 추가. **dr_grpo 본선 중단**(checkpoint-1050, step~1066·33% → GDPO A/B에 자원 집중; ckpt는 Stage-3 init 후보로 병합 대기). GDPO는 초기 requeue로 07-04 09:01 fresh 재시작(step 1까지만 갔던 07-02 런 유실, 체크포인트 전이라 무손실).
- **07-05** — GDPO A/B **step 306/600(51%) 순항**. dr_grpo와 동일구간 비교: **AccMix·총 reward 동급 + FormatThink 우위(+0.04~0.06, 150스텝 지속)**, `zero_std=0`. 문제정의·해결방안 4축 정리 문서(`docs/project_status_2026-07-05.md`) + SFT 콜드스타트/RFT 상세(부록 A) 작성.
- **07-06** — GDPO A/B **step 450/600(75%)**. AccMix·reward 전 구간 dr_grpo 동급 유지, FormatThink는 300스텝대 우위 후 **400대에서 근접 수렴**.
- **07-07** — **GDPO 완주(step600)·최종 판정**: on-policy 판정창 Acc 0.487 vs 0.490, **층화 홀드아웃(N=200) 0.380 vs 0.390** → 둘 다 **동률**(GDPO 미세우위, 노이즈 내). Stage-2 무차별·downside 없음 → **Stage-3용 GDPO 채택 권고**(`46_eval_gdpo_ab.slurm`). 병행: **HealthBench Hard(1000) 측정** — base **0.229** vs 콜드스타트 **0.224**(동률, 콜드스타트 instr +0.044·출력간결화). 타겟 벤치마크 섹션 신설 + 단계별 추적표(②③ 예정). → [상세](stage3_and_eval.md#타겟-벤치마크-healthbench--의료-성능-측정)
- **07-08~09** — **콜드스타트 Ablation Study**(순가치 확정): `base→dr_grpo`·`base→GDPO`(콜드스타트 無) 병렬 step200 완주(`59946`/`59970`). **FormatThink 100스텝 ~0 정체**(콜드스타트 0.26 시작)·clip 40%·저속, 홀드아웃 checkpoint-200 **dr_grpo 0.18/GDPO 0.165**(콜드스타트 SFT 단독 0.22에도 미달, 콜드스타트+RL 0.38의 절반). 2×2 강한 상호작용(RL 이득 콜드스타트 조건부) → **Stage-1 필수 확정**([상세](stage1_coldstart.md#콜드스타트-ablation-study-순가치-확정-2026-07-09), `47_eval_ablation.slurm`).
- **07-13** — **GSPO 홀드아웃 갭 보완**(`48_eval_gspo_holdout.slurm`): trainonly 3종 중 GSPO만 홀드아웃 미측정이었음 → step600 측정 **0.290**. on-policy 동률(0.500)이었으나 홀드아웃 3종 최하위(train-test 격차 −0.21, dr_grpo −0.11의 2배) = **일반화 실패**. → GSPO 미채택 근거 강화(홀드아웃이 on-policy 오판을 교정). DAPO·baseline은 구 데이터(홀드아웃 포함) 학습이라 clean 측정 불가·오염 참고치만(`49_eval_contaminated.slurm`).
- **07-16** — **콜드스타트 v3 착수 (일반+의료 혼합)**. ① **v2 형식 천장 발견**: `format_think` 를 v2 학습데이터에 직접 적용 → **0.473**(구조위반 50.1%·빈 think 4.8%). 원인은 `build_rft_coldstart.py` 의 느슨한 `closed()` 게이트 + "가장 짧은 3개" 선별(객관식 찍기 우선). **RL 형식 0.425 정체는 데이터가 천장이었음**. ② **데이터 전수 스크리닝**(전부 다운로드·실측): 채택 = OpenMedReason 150K(의료·1,927자)·VisualWebInstruct-verified 97K(MIT·933자)·VLAA 79K(이미 형식 1.0). 탈락 = MMFineReason(9~11K자 장황)·S-Chain(정답 3개)·MMK12/MM-Eureka/ThinkLite-VL(trace 없음)·자기수확(4.4K자). ③ **OpenMedReason 정답위치 편향 발견**(4지선다 A=77%·5지선다 A=86% → 찍기로 77~86%) → 문자별 균형 샘플링으로 해소. ④ `build_mixed_coldstart.py` 신규 + `10_sft.slurm` 기본값 교정(폐기된 v1 을 가리키고 있었음). ⑤ **홀드아웃 22% 중복 발견**(이미지 바이트해시 동일 214/972) — 기록만, 재측정 보류. ⑥ **예산 실측: 4,155/5,000(83%)** → Stage-2 재실행 불가 확인.
- **07-17** — 폐기 콜드스타트 빌더(v1/v2 계열) `scripts/_archive/` 로 이동 + 폐기 사유 README 동봉. `10_sft.slurm` 이 폐기된 v1 을 기본값으로 가리키던 문제까지 정리해 **그냥 `sbatch` 하면 v3 가 돌도록** 교정.
- **07-18** — **v3 SFT 학습 완주**(job 66255, 4gpu·41분·298스텝). 9,507건 2 epochs, `train 1.12→0.60`·`eval_loss 0.679→0.668`·`eval_token_acc 0.794`, grad_norm 0.29 안정·**과적합 없음**. 병합+프로브 스크립트(`12_merge_mixed.slurm`) 작성. → [학습곡선](stage1_coldstart.md#v3-학습-결과--학습곡선-job-66255-2026-07-18)
- **07-20** — **v3 홀드아웃 평가 완주**(job 69807, 972건 전량): **strict `format_think` 0.909**(v2 데이터 천장 0.473·RL 정체 0.425 **완파**), **accuracy 0.348**(v2 ~0.22, v2-RL 0.380/0.390 에 근접) — *RL 없이 SFT 만으로* 달성. 층별 math 0.324/vl 0.368, mean 1,982자, 형식위반 9.1%는 2048토큰 잘림. 부수: `sft_mixed_merged` 생성(Stage-2 새 init). **오염 비대칭 규명** — v3 는 DeepVision 미학습이라 홀드아웃 22% 오염 이득이 없어 **비교가 v3 에 보수적**. v2 동일조건 재측정 제출(70342 TIMEOUT → 70671 재제출). → [평가결과](stage1_coldstart.md#v3-홀드아웃-평가-결과--형식-천장-완파-job-69807-2026-07-20)
- **07-21** — **v2 동일조건 재측정 완주**(job 70671, 972건 전량, 같은 하니스). v2-SFT: acc **0.295**·strict `format_think` **0.185**·mean **4,824자**. **A/B 확정**: 형식 v2 0.185 → v3 0.909(**5배**, 압도적), 정답률 v2 0.295 → v3 0.348(**+0.053·+18%**, 전부 **vl** 0.270→0.368; **math 는 0.3245 우연 동률**, per-sample 검증으로 artifact 아님 확인 — 정답 겹침 76/147). 과거 "~0.22"는 소표본값이라 폐기. **간결성**: v3 가 절반 길이(1,982 vs 4,824자)라 v2 는 1차 재측정서 TIMEOUT 났고 RL 잘림→형식0 의 근원. → [평가결과](stage1_coldstart.md#v3-홀드아웃-평가-결과--형식-천장-완파-job-69807-2026-07-20)
- **07-22** — **예산 방향전환 + Stage-2 풀확장 착수**. 다른 계정 5,000 노드시간 확보 → "836으로 Stage-2 vs Stage-3 택1" 제약 해제. v3 약점(math 동률) 겨냥해 **STEM RLVR 추가**: `MMK12`(15,616)·`ThinkLite-VL-hard`(11,031) 다운로드·검증(정답형식 검증가능 확인)·변환(`convert_to_swift` 스키마 2종 추가, CPU 잡). 콜드스타트서 탈락했던 데이터(trace 없음)가 RLVR 엔 적합.
- **07-23** — **Stage-2 확장 세팅 완료·GitHub 공유**. `build_stage2_mix.py`(bytehash dedup, seed42) → train **128,349**/holdout **1,673**(신규 누수 0 검증). `20_rlvr_grpo` 기본값=v3 init+확장셋, `launch_stage2_expanded.sh`, `docs/stage2_expansion_runbook.md`. **[`HANDOFF.md`](../HANDOFF.md) 전면 갱신**(계정 이식 sed·환경함정·실행절차). → 다른 계정이 clone→재현→본실행. → [풀확장](stage2_experiments.md#0-풀확장-재설계-2026-07-2224-데이터--07-28-본실행-착수)
- **07-24** — **의료 데이터 추가 + 확장셋 재조립**. 의료 VQA 스크리닝(전부 실측): Kvasir(58K인데 고유이미지 671·degenerate)·SLAKE(이미지 450)·PathVQA(절반 개방형) 탈락, **PMC-VQA(329K MC·PubMed 광범위·B/C/A/D 균형) 채택**(`build_pmcvqa.py` — CSV+MC+zip 선택추출). reward 견고성 실측(0/12 손실 → 정규화 불필요). ThinkLite 드롭. **재조립 = DeepVision 40K+MMK12 15.2K+PMC-VQA 20K = 74,787**(일반53/math20/의료26), **DeepVision 구 22% 오염까지 최종 해소**(홀드아웃 해시 배제), 전 소스 dedup 누수 0. **GDPO 배선 수정**(런처가 검증된 `21_rlvr_grpo_adv` 경유하도록 — dynamic_sample 코어 누락 버그). `docs/stage2_data.md` 신설. → [상세](stage2_experiments.md#0-풀확장-재설계-2026-07-2224-데이터--07-28-본실행-착수)
- **07-27~28** — **계정 이관 + 환경 복구 + Stage-2 본실행 착수**. ① **k252a02 이관**: `work/` 293G 직접전송(rsync), jsonl 42만곳·스크립트 95곳 경로 일괄치환(이미지 200/200 해석 검증). owner 전용 3파일(`guide.pdf`·`plan.hwp`·`.msc`)은 권한상 미전송. ② **apptainer 파손 발견**(`libsubid.so.3` 부재 + GLIBC_2.28 요구 vs 호스트 2.17; 로그인·계산 노드 공통, 7/21 까지는 정상) → **`ENV_MODE=loader` 우회 구현**: sandbox 안 glibc 2.35 로더로 sandbox python 직접 구동(`runc.sh` + `bin/python` shim). 함정 5개 해결(sys.executable·PYTHONHOME·LD_LIBRARY_PATH 분리·CUDA_HOME 실경로·Triton 용 gcc 10.2.0). ③ **배선 스모크 완주**: 1GPU(job 72844) 2 step → 8GPU(job 72832) 5 step, AccuracyMix 0.31~0.48·FormatThink 0.95~1.0·`frac_reward_zero_std` **0**·~331 s/it. ④ **1 epoch 체인 제출**(job 73312~73315). ⑤ `docs/rlvr_hparams_external.md` 신설(2026 리포트 대조: KL β·그룹크기·temp·에포크 관행).
- **08-02** — **본실행 중간 점검 + 평가 경로 복구**. ① **중간 점검 보고서** 신설([`stage2_run73924_progress.md`](stage2_run73924_progress.md)) — step 650 기준 6패널 곡선(`scripts/plot_train_curves.py`)·구간 대조 11지표. **정확도 보상 정지**(AccuracyMix +1.3%)·**길이 인플레이션**(mean_length +28.1%, clipped +41.4%)·`frac_reward_zero_std` 4.2배. 인프라는 무결(오류 0건). ② **"1 epoch=2,337 step" 오류 발견·전면 정정**: GRPO 의 `per_device_train_batch_size` 는 completion 을 세므로 `÷ num_generations(4)` 누락 → 실제 **1 epoch=9,348 step**, MAX_STEPS=2,337 은 **0.25 epoch**(로그 `epoch=0.067` 로 확증). 문서 6곳·런처 2종 정정. ③ **평가 경로가 통째로 깨져 있었음**: 평가 스크립트 9종이 apptainer 파손(07-27) **이전** 작성이라 `singularity exec` 직접 호출 → 전부 실행 불가(본실행 2.5일간 중간 평가가 못 돈 이유). `00_common.sh` 에 **`run_serve()` 추가**해 loader 포팅, GPU 노드 실동작 검증. 표본 추출 버그도 수정(앞에서 N줄 자르기 → `_source` 균등 층화, 종전에는 확장 홀드아웃에서 전부 deepvision 만 뽑혔음). ④ **중간 홀드아웃 평가 실행**(job 74060, n=300 층화): base **0.2500** → init **0.4533** → trained(step600) **0.4867**. init→trained **+3.34pp p=0.412 유의하지 않음**(base→init 은 +20.33pp p<0.001 유의) → **판정 불가, 추세 재측정 필요**. 의료(pmcvqa) 0.57→0.53 이 유일한 하락. 스냅샷 `_mideval_snap_step{400,500,600}` 확보.

### TODO
- [x] 환경·모델·데이터 확정 + 전체 변환 (DeepVision 103K / medix 51K)
- [x] LoRA 전환(NVLink 없음) · 간결 콜드스타트 · `accuracy_mix`
- [x] Stage-2 baseline 완주 + **A/B 종결(dr_grpo 승자)**
- [x] Stage-3 RaR 보상·judge·**배선 end-to-end 스모크**(유닛 29/29)
- [x] **홀드아웃 정비 + fresh 1 epoch 재학습** (dr_grpo 본선은 33%서 중단, GDPO A/B로 전환)
- [x] **중간 홀드아웃 벤치마크**(RL 25%): init 0.22→trained 0.38(+73%)
- [x] **Stage-2 홀드아웃 확정** → step600서 **~0.38–0.39 포화**(전량 완주 불필요, 조기확정). step600 ckpt = Stage-3 init 후보
- [x] **GSPO A/B 판정** → 판정창 동률 → **dr_grpo 유지**(미채택)
- [x] **GDPO A/B 판정** → 판정창·홀드아웃 **동률**(0.380 vs 0.390) → Stage-2 무차별, **Stage-3용 채택 권고**
- [x] **콜드스타트 Ablation Study** → base→RL(콜드스타트 無) 0.18 붕괴 → **Stage-1 필수 확정**
- [x] **HealthBench 기준선** → base 0.229 / 콜드스타트 0.224 측정(추적표 ①까지)
- [x] **콜드스타트 v2 결함 규명** → 데이터 `format_think` 0.473 = RL 형식 천장
- [x] **데이터 전수 스크리닝·수급** → OpenMedReason·VisualWebInstruct·VLAA 확보·검증(탈락 6종 근거 기록)
- [x] **`build_mixed_coldstart.py`** 신규(게이트 1.0·A편향 제거·정답정규화·질문단위 분할·난이도 층화) + `10_sft.slurm` 기본값 교정
- [x] **콜드스타트 v3 SFT 실행** → `sft_mixed_lora/checkpoint-298` (job 66255, 41분·298스텝, 과적합 없음)
- [x] **v3 DeepVision 홀드아웃 평가** → **acc 0.348 · strict `format_think` 0.909**(천장 0.473 완파). `sft_mixed_merged` 생성 (job 69807)
- [x] **v2 동일조건 재측정** (job 70671, 07-21) → acc **0.295** / `format_think` **0.185**. 과거 소표본 "~0.22" 대체, v3 A/B 확정(+0.053 acc·vl 주도, 형식 5배)
- [ ] **v3 HealthBench Hard** *(보류 — 비용 재검토 후 결정)* — base **0.229** / v2 콜드스타트 **0.224**(둘 다 `n=1000` 정식 실측)와 동일 하니스로 비교 가능하나, **실측 비용이 크다**: 과거 런 `59666`(base) **7:03 on 4gpu ≈ 28 노드시간**, `59691`(v2) **6:53 on 8gpu ≈ 55**. judge(27B)+타깃 동시 서빙이라 `gpu:2` 필수. → 잔여 예산 배분 결정 후 재검토. [추적표](stage3_and_eval.md#단계별-추적--healthbench-hard-n1000)
- [x] **예산 배분 결정** → 다른 계정 5,000h 확보로 제약 해제, **Stage-2 풀확장 + Stage-3 둘 다** 진행
- [x] **Stage-2 풀확장 데이터·검증** → MMK12·**PMC-VQA(의료)** 수급·변환·조립(train **74,787** 일반53/math20/의료26, holdout 1,772 dedup 누수0), GDPO 배선·런북·HANDOFF·`docs/stage2_data.md` 커밋
- [x] **k252a02 이관 + 환경 복구**(07-27~28) → work/ 293G 직접전송·경로 일괄치환, **apptainer 파손 우회(`ENV_MODE=loader`)**, 배선 스모크 완주(1GPU·8GPU)
- [ ] **Stage-2 풀확장 본실행**(진행중) → 2,337 step 체인 job **73924~73927**(=0.25 epoch, 종전 "1 epoch" 표기는 오류 — [정정](stage2_run73924_progress.md#3-epoch-커버리지--계획-전제가-4배-틀렸다)), 중간 체크포인트 평가 후 조기중단 판단
- [ ] **중간 체크포인트 홀드아웃 평가 즉시 실행** ← step 630 까지 **AccuracyMix +0.7%(정지)**, 길이만 +29.8% 팽창. 계속/중단의 유일한 근거. [중간점검](stage2_run73924_progress.md)
- [ ] **Stage-2 확장 배선 스모크 → 본실행** (`launch_stage2_expanded.sh`, 다른 계정 ~70h) ← **다음 임계경로**
- [ ] **확장 결과 평가**(소스별 홀드아웃) → v3(0.348) 대비 STEM 이득 확인
- [ ] **Stage-3 본실행**(`launch_stage3.sh`) → init 교체 ← **계획서 핵심 산출물, 미시작**
- [ ] Stage-2·3 모델 HealthBench 추적표 ②③ 채움
- [ ] (보류) 홀드아웃 이미지해시 기준 재구성 + clean 3종 재측정 — 예산 확보 시
- [ ] (보류) VLAA vg·coco 이미지 별도 수급(Visual Genome·COCO) → 46,969건 복귀
- [ ] (보류) MMFineReason 을 **RL 단계** 데이터로 검토(`pass_rate` 난이도 라벨)

---
