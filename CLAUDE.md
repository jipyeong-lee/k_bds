# CLAUDE.md — 실행 환경 가이드 (KISTI + Jukyung-Yadok B200)

Stage-2 도메인 전문가 3종 RLVR 학습. 두 클러스터에서 실행 가능.
큰 그림·수치 → [`README.md`](README.md) · 계정 이식 → [`HANDOFF.md`](HANDOFF.md) · B200 스크립트 → [`b200/`](b200/).

> ⚠️ **비밀값 금지**: PAT·비밀번호·OTP를 커밋하지 말 것(GitHub push됨). 자격증명은 `$ORCH_PAT` 등 환경변수로.

---

## 0. 두 실행 환경

| | **KISTI K-BDS** | **Jukyung-Yadok (NHN B200)** |
|---|---|---|
| GPU | A100-40GB PCIe, NVLink 없음 | **8× B200 180GB, NVLink** |
| 접속 | `ssh k266a01@kbds.kisti.re.kr` (OTP+PW) | HTTP API (PAT), 대화형 셸 없음 |
| 실행 | Slurm `sbatch` | 세션 생성 → `exec` → job 폴링 (async) |
| 속도(실측) | GRPO ~213 s/it | **GRPO ~16–27 s/it (스모크)** |
| 상태 | 원본 데이터·검증본 보유 | **✅ pmcvqa 스모크 완주(2026-08-14)** |

원본 데이터(`work/`, ~305G)·검증본은 KISTI `k266a01:~/kbds_project`, tar는 `/scratch/migrate_k266_to_gpu/`.

---

## 1. Jukyung-Yadok B200 — 검증된 작동 레시피

### 1.1 접속
- API base `$ORCH_BASE_URL`(별도 전달, self-signed → `curl -k`) · 헤더 `Authorization: Bearer $ORCH_PAT`
- 계정 `gpu-user-1`(role user) · 노드 `gpu-node-1`(8× B200) · **1년 grant 승인** · reservations 비활성 → 세션+exec 직행
- 노드 실행은 [`b200/drive_node.sh`](b200/drive_node.sh)로 감쌈: 세션생성→exec(stdin=`bash -s`)→폴링→반납.
  PAT로 붙으므로 **사람이 직접 실행**(에이전트 자동실행은 종종 차단됨).

### 1.2 저장소 (관리자 설정 완료)
| 경로 | 성격 | job |
|---|---|---|
| `$ORCH_HOME` (`/NHNHOME/orch-data/users/<uid>`) | 영속 xfs **3.2T** | **R/W** ✅ |
| `$ORCH_DATASETS`/`$ORCH_SHARED` | 영속, admin 관리 | 읽기전용 |
| `$ORCH_SCRATCH` / `/tmp`(tmpfs,noexec) | 세션 | R/W, **휘발성** |

**venv·데이터·체크포인트는 전부 `$ORCH_HOME`에** → 세션 간 재사용. `HOME=/root`는 쓰기 불가라 스크립트가 `HOME`·캐시를 `$ORCH_HOME`로 재지정.

### 1.3 데이터 반입 (KISTI → 플랫폼)
- `POST /me/data/upload` 실측 **~138 MB/s**(저지연 4ms). **단일 요청 60초 제한** → ≤3.5G 청크로 split([`b200/upload_chunked.sh`](b200/upload_chunked.sh)), 노드에서 `cat …*.part | tar xf -`로 재조립.
- **HF 대용량 CDN 차단**(cdn-lfs 000, xethub 403) → HF 직접 다운로드 불가, 반드시 업로드.
- job은 `$ORCH_HOME/uploads`를 파일시스템으로 직접 읽음(API pull 불필요).

### 1.4 환경 스택 (KISTI 검증본 핀) + B200 5대 함정
[`b200/node_setup_and_smoke.sh`](b200/node_setup_and_smoke.sh)가 `$ORCH_HOME/.venv`에 1회 설치(멱등):
```
torch==2.10.0+cu129 torchvision==0.25.0+cu129   # --index-url .../whl/cu129 (Blackwell OK)
vllm==0.19.1 ms-swift==4.1.3 transformers==5.6.2 trl==0.29.1 peft==0.19.1
accelerate==1.13.0 datasets==3.6.0 pyarrow==23.0.1 pillow   # --index-strategy unsafe-best-match
qwen_vl_utils==0.0.14 decord==0.6.0 av==17.0.1              # Qwen-VL, ms-swift가 자동설치 안 함
nvidia-cutlass-dsl==4.5.0.dev0  (--prerelease=allow)        # ← 아래 함정②
math_verify==0.9.0 latex2sympy2_extended==1.11.0 word2number==1.1  # 보상 플러그인
```
**B200에서 뚫어야 했던 함정 (전부 스크립트에 반영됨):**
1. **uv 인덱스**: cu129 extra-index + pypi 혼합 → `--index-strategy unsafe-best-match` 필수.
2. **cutlass-dsl**: pip 최신 4.7.0은 `cute.core.ThrMma` 제거 → vLLM ViT의 quack 깨짐. **4.5.0.dev0으로 다운그레이드**.
3. **Triton 캐시**: `/tmp`가 tmpfs+noexec → 컴파일 `.so` map 실패. `TRITON_CACHE_DIR`·`TORCHINDUCTOR_CACHE_DIR`·`TMPDIR`를 `$ORCH_HOME`(exec)로.
4. **flashinfer nvrtc.h**: vLLM이 Blackwell 커널을 시스템 nvcc(CUDA13)로 JIT 컴파일하는데 `/usr/local/cuda`에 `nvrtc.h` 없음(그 include는 읽기전용). 헤더·lib는 venv(`nvidia/cuda_nvrtc`)에 있으므로 **`CPATH`/`LIBRARY_PATH`/`LD_LIBRARY_PATH`로 노출**(1회 컴파일 후 `$ORCH_HOME`에 캐시).
5. **flash-attn 없음**: 노드에 nvcc는 있으나 dev 헤더 불완전 → 소스빌드 지양. 학습은 **`--attn_impl sdpa`**. (vLLM 롤아웃은 flashinfer가 처리.)

### 1.5 학습 실행
```
ORCH_PAT=pat_xxx bash b200/drive_node.sh b200/node_setup_and_smoke.sh 1 2400
```
스모크 통과 실측(2026-08-14, pmcvqa 3 step, 1 GPU): `SWIFT_EXIT=0`, reward 1.2→0.575,
AccuracyMix/FormatThink/SoftOverlong 정상, mem **172/180 GiB**, ~40 s/it.
3-arm 확장 → [`HANDOFF.md`](HANDOFF.md) §다음.

---

## 2. KISTI K-BDS (원본)
- `ssh k266a01@kbds.kisti.re.kr`(PW+OTP, 대화형). 최초 1회 후 ControlMaster 8h 재사용.
- `~/kbds_project`, 데이터 `work/`(~305G, git 제외). 상세 [`HANDOFF.md`](HANDOFF.md)·[`README.md`](README.md).

## 3. 핵심 식별자
| 항목 | 값 |
|---|---|
| 플랫폼 계정 / 노드 | `gpu-user-1` / `gpu-node-1` (8× B200) |
| init 모델 | `sft_mixed_merged` (Qwen3.5-9B 병합, 18G) |
| 도메인 데이터 | domains/stage2_{deepvision 40k / mmk12 15k / pmcvqa 20k}.jsonl |
| 학습 프레임워크 | ms-swift 4.1.3 (recipe=stable, dr_grpo, GDPO, num_gen=8) |
