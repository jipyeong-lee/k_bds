# B200 플랫폼 사용법 — 8장 LoRA 데모

HuggingFace 에서 모델을 받아 **B200 8장으로 LoRA 튜닝**하는 최소 예제. 실제로 돌려 검증한 코드다.

```bash
bash b200/demo/run_demo.sh      # 로컬에서 실행. 5~8분(첫 실행은 venv 설치로 +3분)
```

## 실측 결과 (2026-08-21)

```
[demo] world_size=8 · model=Qwen/Qwen2.5-0.5B-Instruct
[demo] rank 0~7 → NVIDIA B200 (178 GiB)
trainable params: 2,162,688 || all params: 496,195,456 || trainable%: 0.4359
[demo] epoch 1/3  loss 5.1114
[demo] epoch 2/3  loss 4.4451
[demo] epoch 3/3  loss 3.9956
[demo] 어댑터 저장 → …/demo_run/lora_demo_out  (8.3 MiB)
```

---

## 1. 플랫폼의 작동 모델

이 플랫폼에는 **대화형 셸이 없다.** SSH 로 붙어서 명령을 치는 게 아니라, HTTP API 로 세션을 열고
명령을 던지고 결과를 폴링한다. 그래서 사용 흐름이 항상 다섯 단계다.

| 단계 | 엔드포인트 | 비고 |
|---|---|---|
| ① 코드 업로드 | `POST /me/data/upload?path=<디렉터리>` | `-F "file=@..."` · `path` 는 **쿼리 파라미터**다 |
| ② 세션 생성 | `POST /nodes/<node_id>/sessions` | `{"gpu_count":8}` |
| ③ 명령 실행 | `POST /sessions/<sid>/exec` | **비동기** — `job_id` 를 즉시 돌려준다 |
| ④ 폴링 | `GET /jobs/<job_id>` | `running` → `succeeded`/`failed`/`killed` |
| ⑤ 세션 반납 | `DELETE /sessions/<sid>` | **안 하면 GPU 를 계속 점유한다** |

`gpu_count` 는 0도 된다. **GPU 0장 세션은 학습이 8장을 다 잡고 있어도 열리므로**, 파일 정리·조사
같은 작업을 학습 중에 할 수 있다.

## 2. 세 가지 파일이 하는 일

| 파일 | 실행 위치 | 역할 |
|---|---|---|
[`run_demo.sh`](run_demo.sh) | 로컬 | ①~⑤ 전 과정 + 결과 회수 |
[`payload.sh`](payload.sh) | 노드 | venv 준비 + `torchrun` 8장 |
[`lora_demo.py`](lora_demo.py) | 노드 (8 프로세스) | 실제 학습 |

`torchrun --nproc_per_node=8` 이 GPU 당 프로세스를 하나씩 띄우고, 각 프로세스가 모델 사본을
하나씩 들고 그래디언트만 합친다(DDP). LoRA 라 학습되는 파라미터는 전체의 **0.44%** 뿐이고,
저장되는 어댑터도 8.3 MiB 로 원본(약 1 GB)보다 훨씬 작다.

## 3. 반드시 알아야 하는 함정 다섯 개

데모 코드 절반이 이걸 피하는 데 쓰인다. **모르고 시작하면 전부 한 번씩 밟는다.**

### ⓞ 업로드한 디렉터리에는 job 이 쓸 수 없다
업로드는 **API 서비스**가 파일을 만들고, 학습은 **job 컨테이너 사용자**가 돌린다. 소유자가 달라
`demo/` 안에 로그를 쓰려 하면 `Permission denied` 다. → 코드는 `demo/` 에서 **읽고**, 출력은 job 이
직접 `mkdir` 한 `demo_run/` 에 **쓴다.** (같은 이유로 `uploads/` 는 세션에서 `rm` 도 안 된다.)

### ① `HOME=/root` 가 쓰기 불가
HF·pip 캐시가 만들어지지 않는다. → `export HOME="$ORCH_HOME"`

### ② `/tmp` 가 tmpfs + noexec
Triton 이 컴파일한 `.so` 를 map 하지 못해 죽는다. → `TMPDIR`·`TRITON_CACHE_DIR`·
`TORCHINDUCTOR_CACHE_DIR` 을 `$ORCH_HOME` 아래로

### ③ 세션 venv 는 세션과 함께 사라진다
세션마다 `.venv` 가 새로 생기므로 매번 재설치하게 된다. → `$ORCH_HOME/.venv-demo` 에 만들면
다음 실행은 즉시 시작된다

### ④ torch 는 cu129 휠이어야 한다
Blackwell(B200)은 일반 pypi 휠로 못 쓴다. → `--index-url https://download.pytorch.org/whl/cu129`

### ⑤ flash-attn 이 없다
노드에 nvcc 는 있지만 dev 헤더가 불완전해 소스빌드는 지양한다. → `attn_implementation="sdpa"`

## 4. 결과는 화면이 아니라 파일로 받는다

job 은 `timeout_sec` 에 도달하면 무조건 `killed` 되고, **그때 `stdout_tail` 이 통째로 빈다.**
그래서 로그를 파이프로 흘리지 말고 `$ORCH_HOME` 의 파일에 남긴 뒤 파일 API 로 받는다.

```bash
GET /me/data?path=<디렉터리>       # 목록 (JSON)
GET /me/data/file?path=<파일>      # 내용
```

**이 두 API 는 세션이 없어도 동작한다** — 학습이 8장을 다 쓰는 동안에도 진행 상황을 읽을 수 있다.

> ⚠️ **`GET /me/data/file` 은 safetensors 를 못 받는다** — 내용 때문에 HTTP 500 이 난다
> (크기·확장자 무관, 2026-08-21 재확인). 모델 가중치를 꺼내야 하면 **`tar` 로 감싸서** 받을 것.
> 자세한 내용과 재귀 다운로더는 [`../pull_all.py`](../pull_all.py) · `CLAUDE.md` §1.5 참고.

## 5. 자격증명

저장소 루트 `.env` (gitignore 됨, 형식은 [`.env.example`](../../.env.example)):

```
ORCH_BASE_URL=https://web-orchestration.koreahealth.ai
ORCH_PAT=pat_...
```

PAT 하나로 모든 API 가 된다. 재발급은 웹 UI `/ui/users.html` → Access tokens
(평문은 생성 시 1회만 표시된다).

## 6. vLLM 자체 호스팅 인퍼런스

[`vllm_test.sh`](vllm_test.sh) 로 두 경로를 검증했다(2026-08-21, `vllm==0.19.1` · `tp=2`).

```
[A] 오프라인 배치   3건 · 88 토큰 · 3.63s · 24.2 tok/s   (모델 로드 45s)
[B] OpenAI 호환 서버 /health OK · /v1/chat/completions 정상 · /v1/models 조회 정상
```

> ⚠️ **외부에서는 이 서버를 부를 수 없다.** 플랫폼 API 에 포트 노출·ingress 기능이 없어
> 노드의 8000 포트는 job 안에서만 보인다. 배치 추론(job 안에서 호출 → 결과를 파일로)은 되지만,
> 상시 서비스로 쓰려면 별도 인프라가 필요하다.

vLLM 특유의 함정 셋:

| | 내용 |
|---|---|
| stdin 금지 | 코드를 heredoc 으로 넘기면 워커가 메인 모듈을 다시 열다 `FileNotFoundError('<stdin>')` 로 죽는다 → **파일로 써서** 실행 |
| `__main__` 가드 필수 | `tp>1` 이면 spawn 워커가 스크립트를 **다시 실행**한다(`_fixup_main_from_path`). 최상단에 `LLM(...)` 이 있으면 자식이 또 엔진을 만들려다 죽는다 |
| `tp` 는 헤드 수의 약수 | Qwen2.5-0.5B 는 14 헤드라 1·2·7·14 만 된다 — **8은 안 된다.** 8장을 다 쓰려면 헤드가 8의 배수인 모델이어야 한다 |

## 6. 바꿔볼 것

| 목적 | 방법 |
|---|---|
다른 모델 | `DEMO_MODEL=Qwen/Qwen2.5-7B-Instruct bash run_demo.sh` |
실제 데이터셋 | `lora_demo.py` 의 `PAIRS` 를 `datasets.load_dataset(...)` 으로 교체 |
GPU 수 조절 | `payload.sh` 의 `--nproc_per_node` 와 `run_demo.sh` 의 `gpu_count` 를 같이 바꿀 것 |
긴 학습 | `TIMEOUT=14400 bash run_demo.sh` (14,400초까지 완주 실측) |

**긴 학습에서는 체크포인트를 자주 저장할 것.** job 은 `timeout_sec` 에 죽고 마지막 저장 이후
진행분은 버려진다 — 그 손실량은 난수가 아니라 `((T-셋업)/step시간) mod save_steps` 로 결정된다.
