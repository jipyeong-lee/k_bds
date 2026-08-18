# CLAUDE.md — 실행 환경 가이드 (KISTI + Jukyung-Yadok B200)

Stage-2 도메인 전문가 3종 RLVR 학습. 두 클러스터에서 실행 가능.
큰 그림·수치 → [`README.md`](README.md) · 계정 이식 → [`HANDOFF.md`](HANDOFF.md) · B200 스크립트 → [`b200/`](b200/).

> ⚠️ **비밀값 금지**: PAT·비밀번호·OTP를 커밋하지 말 것(GitHub push됨). 자격증명은 `$ORCH_PAT` 등 환경변수로.

> ⚠️ **셸 출력 필터(rtk PreToolUse 훅)**: 명령 출력이 재작성·압축된다(2026-08-18 이 저장소 실측).
> `git status` 는 git 출력이 아니라 `~ Modified: 1 files` 형태의 **요약본**이 오고, 긴 출력은
> `// ... N more lines` 로 접힌다 — §1.5 의 "stdout 8KB 절단" 과 같은 부류의 오판 위험이다.
> **트리 판단은 `git status --porcelain`**(그대로 통과) · 긴 지표·로그는 파일로 받아서 읽을 것 ·
> 패치는 `git diff --output=x.patch` 로 git 이 직접 쓰게 할 것.
> (`git diff` 가 `git apply` 못 읽는 형태로 깨지는 사례가 `new_jarvis` 에 실측 기록돼 있다.
> 이 저장소에선 재현되지 않았으나 예방 형식을 쓴다. `grep` 은 여기선 `/usr/bin/grep` 과 결과 동일.)

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
# 스모크 1회 (pmcvqa 3 step) — 환경 검증용
ORCH_PAT=… ORCH_BASE_URL=… bash b200/drive_node.sh b200/node_setup_and_smoke.sh 8 2400
# 본실행: job 이 timeout 마다 끊기므로 체크포인트 재개 체인으로 돈다
nohup bash b200/chain_epoch.sh deepvision 7200 &   # 내부에서 run_epoch.sh 를 반복 투입
bash b200/release_session.sh                        # 세션 조회 / --all-gpu 로 강제 반납
```
스모크 통과 실측(2026-08-14, pmcvqa 3 step, 1 GPU): `SWIFT_EXIT=0`, reward 1.2→0.575, ~40 s/it.

**플랫폼 운영 함정 (2026-08-15 실측, 전부 겪은 것):**
| | 내용 |
|---|---|
| job 수명 | `timeout_sec` 이 그대로 상한. 3600 을 주면 3600 초에 `killed`, **7200·14400 모두 완주 실측**(60분 고정 아님. 14400 은 2026-08-18: `NODE_START 18:34:01` → `NODE_END 22:34:20` = 14,419 초). job 이 죽으면 마지막 체크포인트 이후 진행분이 버려지는데 **그 양은 난수가 아니라 결정론적**이다 — job 길이도 step 시간도 고정이라 `((T-90)/step시간) mod save_steps` 가 그대로 손실이 된다. 7200 은 36.3 step 진행 → 4.3 버림(12 job 평균 실측 2.9), 14400 은 74.5 → 10.5 버림(실측 11). **그래서 14400 은 실효 이득이 없었다**: 225.9 s/step(64 step / 14,456 초) vs 7200 의 226.5. 균등난수라면 12 job 평균이 7.5 여야 하는데 2.9 인 점이 결정론의 증거다. 이득을 보려면 T 를 더 늘려 **버려지는 절대량(0~15 step)의 비중 자체를 낮추는 것**이 유일하게 견고하다 — 16의 배수 step 에 맞추는 정밀 조준은 step 시간이 191~204 로 흔들려 못 쓴다 |
| killed 시 | **stdout_tail 이 통째로 빈다.** 결과는 반드시 `$ORCH_HOME` 파일에 남기고 별도 짧은 job 으로 읽을 것 |
| 세션 반납 | 폴링 프로세스가 죽으면 `DELETE` 가 실행되지 않아 **세션이 GPU 를 계속 점유** → 이후 job 이 전부 `session create failed`. 조회·해제는 `/sessions` (`/nodes/<id>/sessions` 는 405) |
| 동시 조회 | 학습이 GPU 8장을 잡는 동안 **추가 세션이 안 열린다**(`session create failed`) → 노드 exec 로는 job 교체 틈에서만 조회 가능. **단 파일 읽기는 세션이 필요 없다** ↓ |
| **세션 없는 파일 읽기** | `GET /me/data/file?path=<$ORCH_HOME 기준 상대경로>` 가 파일 내용을 그대로 준다 — 학습 중에도 언제든. `/me/data?path=<dir>` 는 목록, `download`·`cat` 는 404. **stdout 8KB 절단도 없다** → 로그를 통째로 받아 전 step 파싱. [`b200/pull_file.sh`](b200/pull_file.sh) → [`b200/parse_log.py`](b200/parse_log.py) → [`b200/plot_progress.py`](b200/plot_progress.py) |
| 체크포인트 | ms-swift 는 `$OUT/v<N>-<날짜>/checkpoint-<step>` 에 저장. `$OUT/checkpoint-*` 로 찾으면 **영영 못 찾아 매 job 이 step 0 부터 재시작** |
| **rollout 포트** | 앞 job 이 killed 되면 그 소켓이 **TIME_WAIT** 로 남고, vLLM 은 실패하지 않고 **조용히 8001 로 뜬다.** health 가 8000 만 보면 서버가 53초 만에 멀쩡히 떠 있는데도 타임아웃 사망(교체마다 10~30분 손실). TIME_WAIT 는 "연결이 되는가"로는 감지할 수 없다(연결은 실패하고 bind 만 실패) → 포트를 비우려 하지 말고 로그의 `Uvicorn running on ...:<포트>` 를 읽어 **그 포트를 따라갈 것** |
| **chain 페이로드 고정** | `chain_epoch.sh` 는 루프 **시작 시 1회만** `run_epoch.sh` → `run_epoch_<arm>.sh` 를 만든다. 체인 도중 `run_epoch.sh` 를 고쳐도 **다음 job 에 반영되지 않는다** → 페이로드도 같이 갱신할 것 |
| **stdout 8KB** | 노드 exec 의 stdout 은 8KB 근처에서 **앞부분이 잘린다**. 289 행 CSV 가 191~283 만 돌아왔다. 솎아내면 **최저점이 사라져 오판한다**(실제로 `zero_std` 최대 0.357 을 "전 구간 0" 으로 읽었다) → 지표는 솎지 말고 위 파일 API 로 통째로 받을 것 |
| **DELETE 는 거짓말한다** | `DELETE /me/data?path=…` 는 **`runs/` 아래에서 무조건 `{"removed":true}` 를 주고 아무것도 안 지운다**(2026-08-18 실측, 334 파일). 경로 4형태(`a/b`·`./a/b`·`/a/b`·`%2F`)·`recursive=true` 전부 동일, `DELETE /me/data/file` 은 405. 실제로 지워지는 곳은 **`$ORCH_HOME` 직하 파일과 `uploads/` 아래**뿐이다 — API 서비스가 직접 만든 파일만 unlink 되고, 학습 job 이 만든 파일은 실패를 삼키고 200 을 준다. **`runs/` 정리는 GPU 세션 `exec` 의 `rm -rf` 로만 가능** → 학습 중엔 세션이 안 열리니 arm 교체 틈에 할 것. 200 을 삭제 증거로 쓰지 말고 반드시 재조회로 확인할 것 |

**배치·메모리 실측 (deepvision, max_completion 8192):**
- `PDTBS 2 × ACCUM 8 × world 7 = 112` 가 상한(171 s/step). PDTBS 4 는 235 GiB 가 필요해 thrashing
- 메모리는 3 step 117.5 GiB → 41 step **170.6/180 GiB** 로 상승 → `PYTORCH_ALLOC_CONF=expandable_segments:True` 필수
- **`gradient_checkpointing=false` 는 불가** — 16K 시퀀스에서 PDTBS 1 조차 OOM(178.35 GiB 중 357 MiB 잔여)
- 순차 롤아웃은 191 s/step(롤아웃 287s + 학습 286s = **GPU 절반이 절반 시간 유휴**)
  → **`--async_generate true`** 로 오버랩. `vllm_mode server` 전용이라 colocate 로는 못 쓴다.
  ms-swift 4.1.3 에 이미 있다(`--vllm_enable_prefix_caching`·`--steps_per_generation`·`--num_iterations` 도 동일)

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
| 학습 프레임워크 | ms-swift 4.1.3 · `loss_type=dr_grpo` + `scale_rewards=gdpo` + `async_generate=true` · num_gen 16 |
| off-policy 보정 | `rollout_importance_sampling_mode=token_truncate` (threshold 2.0) · `beta=0` · `temperature=1.0` |
| 버전 | 최신은 4.5.0 이지만 **mismatch 관련 인자가 4.1.3 과 동일** → 업그레이드 이득 없음(인자 diff 확인) |
