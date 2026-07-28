# 기술 레퍼런스 — 환경·모델·학습방식·보상·논문

> 이 문서는 [`README.md`](../README.md) 에서 분리된 상세 기록입니다. 요약·현황은 README, 상세는 여기.

## 기술 레퍼런스

### 환경: Singularity 컨테이너 (확정·검증 완료)
- 노드 OS **CentOS 7.9 / glibc 2.17** → 최신 ML 패키지(특히 vLLM·xformers) pip/conda 설치 불가 → **공식 ms-swift 컨테이너**.
- 이미지: `...modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.1-modelscope1.35.4-swift4.1.3` (Gemma4 지원=swift≥4.0.4).
- **계산노드 검증**(드라이버 550=CUDA12.4): swift4.1.3 / torch2.10+cu129 / vllm0.19.1(`vllm._C` OK) / transformers5.6.2. CUDA **마이너 호환**(12.9 빌드를 12.4 드라이버) 정상.
- SIF는 실행마다 추출이 느려 **sandbox(디렉토리)** 변환 사용(`work/images/ms-swift-413-sandbox`).
- 🚨 **2026-07-27~ apptainer 파손 → `00_common.sh` 기본값이 `ENV_MODE=loader`**(구 기본값 `container`). 클러스터 공용 apptainer 1.4.5 가 `libsubid.so.3` 부재 + GLIBC_2.28 요구(호스트 2.17)로 실행 불가 — 로그인·계산 노드 공통, **이미지는 정상**이라 재빌드는 무의미(빌드도 불가). 우회는 sandbox 안 **glibc 2.35 로더**로 sandbox python 을 직접 구동(`runc.sh` + `bin/python` shim). GPU 검증: torch2.10+cu129 `cuda_avail True`·vllm0.19.1·GRPO 8GPU 5 step 완주(job 72832). **apptainer 복구 시 `ENV_MODE=container` 로 원복**. → [`HANDOFF.md`](../HANDOFF.md) §3
- 최신 swift 4.2.3은 CUDA 13 요구라 이 드라이버서 불가. 대안 이미지: swift3.8.3/cuda12.6.3, swift3.6.4/cuda12.4.0.

### 베이스 모델 & ms-swift 4.x 주의
- **`Qwen/Qwen3.5-9B`**(멀티모달, `MODEL_TYPE=qwen3_5`). config `qwen3_5`/processor `Qwen3VLProcessor`. 게이트 없음, `work/hf_cache` 캐시(`HF_HUB_OFFLINE=1` 오프라인).
- ❌ **Gemma 4 12B 불가**: 실제 `model_type=gemma4_unified` + transformers 5.10.dev + CUDA 13 이미지 필요 → 드라이버 550서 구동 불가(다운로드본 정리).
- ⚠️ **swift 3.x→4.x 인자 변경**: `--train_type`→**`--tuner_type`**, `--reward_funcs_plugin X` 폐지→**`--reward_funcs ... X`** 직접 나열. 보상 plugin은 `swift.rewards`(`ORM`/`AsyncORM`/`orms`) API.

### 학습 방식: LoRA (하드웨어 제약 대응)
- **NVLink 없음**(실측): 8gpu = A100 80GB **PCIe**, `nvidia-smi topo -m` 전부 PHB → GPU P2P 차단 → NCCL **SHM(호스트 RAM) 폴백**.
- 결과: **full-FT 멀티GPU는 18GB gradient all-reduce 병목** → 9B·짧은 시퀀스인데도 **375~660초/step**(하이퍼바이저 ACS라 수정 불가).
- **대응 = 전 단계 LoRA**: adapter grad(수십 MB)만 통신 → **~128s/step(GRPO)·~5s/step(SFT)** = ~5~75배↑. base 동결이라 **DeepSpeed 없이 DDP** 충분.
  - cold-start LoRA → `swift export --merge_lora`로 병합 → 다음 단계 `INIT_MODEL`.
  - LR 기본: SFT lora `1e-4`/full `1e-5`, GRPO lora `1e-5`/full `1e-6`. `LR=` override.

### 보상 설계 (Stage-2)
- **출력 형식**: `<think>간결 추론</think><answer>최종답</answer>` (`\boxed{}` 미사용).
- **`accuracy_mix`**(`configs/accuracy.py`): 내장 `accuracy`는 객관식 letter("B")·기호 미파싱 → DeepVision ~48%가 letter라 보상 절반 소실 → 커스텀 **수식=math_verify / letter·문자열=정규화일치** 분기. 가중치 `accuracy_mix 1.0 : format_think 0.2 : soft_overlong 0.2`.
- **vLLM colocate**: `--vllm_mode colocate`·`--vllm_mm_processor_cache_gb 0`(mm_hash AssertionError 회피)·`--sleep_level 1`.

### 논문 레퍼런스 (전 링크 arXiv 원문 검증 완료)

| 기법 | 논문 제목 | 저자·발표 | arXiv |
|------|------|------|------|
| **GRPO** | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | DeepSeek, 2024-02 | [2402.03300](https://arxiv.org/abs/2402.03300) |
| **DAPO** | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | ByteDance, 2025-03 | [2503.14476](https://arxiv.org/abs/2503.14476) |
| **Dr.GRPO** | Understanding R1-Zero-Like Training: A Critical Perspective | Sea AI Lab, 2025-03 | [2503.20783](https://arxiv.org/abs/2503.20783) |
| **GSPO** | Group Sequence Policy Optimization | Alibaba/Qwen, 2025-07 | [2507.18071](https://arxiv.org/abs/2507.18071) |
| **GDPO** | GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL | NVIDIA, 2026-01 | [2601.05242](https://arxiv.org/abs/2601.05242) |
| **RaR** (Stage-3) | Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains | 2025-07 | [2507.17746](https://arxiv.org/abs/2507.17746) |

#### 각 논문: 무슨 문제를 → 어떤 방법으로 (원문 요약)

시간순 계보 **GRPO(기반) → DAPO/Dr.GRPO(GRPO 결함 보수) → GSPO/GDPO(직교 확장) → RaR(검증불가 도메인 확장)**. 각 논문이 **직전 방법의 어떤 한계**를 겨냥했는지가 핵심:

| 논문 | 겨냥한 문제 (직전 방법의 한계) | 도입한 방법 (해결책) |
|------|------|------|
| **GRPO** | PPO는 정책망과 맞먹는 크기의 **가치망(critic)**을 따로 학습 → 메모리·연산 부담↑, 토큰별 가치추정 불안정 | critic **제거**. 한 프롬프트에 여러 응답(그룹)을 뽑아 **그룹 내 보상 상대값**(평균 빼고 ÷std)을 advantage로 → 그룹평균이 baseline 역할, 메모리↓·안정 |
| **DAPO** | 대규모 LLM RL의 **엔트로피 붕괴**(조기수렴)·**후반 gradient 소실**·**길이 폭주/잘림 노이즈** + SOTA 레시피 비공개 | **4기법**: Clip-Higher(탐색 보존→붕괴 억제) · Dynamic Sampling(std=0 그룹 폐기·재샘플) · Token-level PG Loss(길이 정규화 편향 제거) · Overlong Shaping(초과길이 패널티/필터) |
| **Dr.GRPO** | GRPO의 **두 최적화 편향**: ① ÷시퀀스길이 → **응답 길이 편향**(오답이 길어짐) ② ÷그룹std → **난이도 편향**(문제 가중 왜곡) | 두 정규화 **삭제** — 손실은 길이 아닌 **상수 정규화**, advantage **÷std 제거** → unbiased 추정, 토큰효율↑·성능 유지 |
| **GSPO** | GRPO/PPO의 **토큰 단위 IS**는 긴 시퀀스에서 비율 **분산 누적**·클립 증폭 → 불안정(MoE·장문 붕괴) | IS를 **시퀀스(응답) 단위**로 정의(시퀀스 우도 비율·길이 정규화) + 클립·보상·최적화도 시퀀스 단위 → 분산 억제, MoE 안정(Qwen3) |
| **GDPO** | **다중 보상** 가중합 후 통짜 정규화 → 서로 다른 조합이 **같은 advantage로 붕괴(collapse)**, 보상 간 상대차 소실 | 결합 **전에 각 보상 함수를 그룹 내 개별 정규화(z-score)** 후 결합 → 각 보상 상대크기·특성 보존(멀티리워드 균형) |
| **RaR** (Stage-3) | RLVR은 수학·코딩 등 **검증 명확 도메인**엔 강하나, 의료·과학처럼 **다기준 미묘 판단**이 필요한 개방형 도메인엔 부적합 | **루브릭 체크리스트**를 구조화 보상으로 — judge가 기준별 채점 후 **가중 집계해 복합(부분점수) 보상** → 총체적 품질 최적화 (원논문=인스턴스별, 우리=**정적 통일** 채택) |

> 우리 파이프라인 대응: **Stage-2 = Dr.GRPO 채택**(길이·난이도 편향이 우리 plateau의 직접 원인) + GSPO/GDPO clean A/B 추가검증 · **Stage-3 = RaR**(의료 VQA=검증불가 도메인). 채택 근거·A/B 결과는 [Stage-2 통합 비교](stage2_experiments.md#2-기법-통합-비교--grpo-계열-5종-clean-ab) · [Stage-3](stage3_and_eval.md#stage-3--의료-rl-rar-루브릭-보상) 참조.

---
