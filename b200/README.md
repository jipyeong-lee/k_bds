# b200/ — Jukyung-Yadok B200 실행 스크립트

NHN B200 플랫폼(8× B200)에서 Stage-2 학습을 돌리는 검증된 스크립트 묶음.
배경·함정·저장소 모델은 [`../CLAUDE.md`](../CLAUDE.md) §1 참조.

## 전제
```bash
export ORCH_PAT=pat_xxxxx          # 플랫폼 PAT (커밋 금지)
export ORCH_BASE_URL=https://...   # 플랫폼 API base (커밋 금지 — 기본값 없음, 미설정 시 즉시 중단)
```

## 파일
| 파일 | 역할 | 실행 위치 |
|---|---|---|
| `drive_node.sh` | 노드에 payload 스크립트를 exec (세션→job→폴링→반납) | 로컬/게이트웨이 |
| `node_setup_and_smoke.sh` | `$ORCH_HOME`에 venv·데이터 셋업(멱등) + pmcvqa 3-step 스모크 | 노드(payload) |
| `upload_chunked.sh` | KISTI tar를 3.5G 청크로 split 후 `/me/data/upload` | KISTI |
| `make_tars.sh` | KISTI에서 업로드용 tar 생성(code/model/data/pmcvqa) | KISTI |
| **`run_epoch.sh`** | arm 하나를 1 epoch 학습. 체크포인트 자동 재개 + 붕괴 감시자 | 노드(payload) |
| **`chain_epoch.sh`** | job 이 timeout 마다 끊기므로 `run_epoch.sh` 를 재투입하는 루프 | KISTI |
| **`check_progress.sh`** | 진행 step·보상·메모리·off-policy 지표 조회 | 노드(payload) |
| **`bench_pdtbs.sh`** | GEN_BATCH 고정 후 PDTBS 만 올려 학습 처리량 상한 측정 | 노드(payload) |
| **`release_session.sh`** | 반납 안 된 세션 조회/해제 (`/sessions`) | KISTI |
| **`pull_file.sh`** | `$ORCH_HOME` 파일을 **GPU 세션 없이** 내려받는다 (`/me/data/file`) | 로컬/KISTI |
| **`parse_log.py`** | 받은 학습 로그 → CSV **전 step** (솎지 않는다) | 로컬 |
| **`plot_progress.py`** | CSV → 학습 곡선 PNG 6 패널 (`../README.md` 에 실린다) | 로컬 |
| `dump_metrics.sh` | (구) 노드 안에서 로그 → CSV. stdout 8KB 때문에 80 행으로 솎아야 했다 | 노드(payload) |
| `dump_collapse.sh` | (구) 붕괴 구간만 진단 열 15개로. 위 3종으로 대체됨 | 노드(payload) |

## 순서
```bash
# 1) (KISTI) 업로드용 tar 생성
bash b200/make_tars.sh
# 2) (KISTI) 청크 업로드 → 플랫폼 $ORCH_HOME/uploads
ORCH_PAT=$ORCH_PAT bash b200/upload_chunked.sh /scratch/migrate_k266_to_gpu
# 3) 노드에서 셋업+스모크 (venv·데이터는 이후 세션 재사용)
ORCH_PAT=$ORCH_PAT bash b200/drive_node.sh b200/node_setup_and_smoke.sh 8 2400
# 4) 1 epoch 본실행 — job 이 timeout 마다 죽으므로 체인으로 돈다
nohup bash b200/chain_epoch.sh deepvision 7200 > chain_deepvision.log 2>&1 &
```

## 1 epoch 체인이 필요한 이유
job 은 `timeout_sec` 에 도달하면 `killed` 되고 **그때 stdout 이 통째로 사라진다.** 그래서
`run_epoch.sh` 는 ① `--max_steps` 를 **누적 목표**(1 epoch 전체)로 주고 — job 마다 쪼개면 lr
스케줄러가 매번 다시 깔려 몇 step 만에 0 으로 소멸한다 — ② `$OUT/v<N>-<날짜>/checkpoint-*` 에서
최신 체크포인트를 찾아 재개한다. `chain_epoch.sh` 는 "죽으면 다시 넣는다"만 한다.

**노드 exec 로** 하는 진행 확인은 job 교체 틈에서만 가능하다(학습이 GPU 8장을 잡으면 추가 세션이 안 열린다).
그래서 `chain_epoch.sh` 가 매 job 종료 후 `check_progress.sh` 를 자동으로 한 번 돌려 로그에 남긴다.
**파일 읽기는 세션이 필요 없으므로** 곡선 갱신은 학습 중에도 언제든 된다:
```bash
bash b200/pull_file.sh train_deepvision_ep1_gdpo_async_tis.log /tmp/train.log
python3 b200/parse_log.py /tmp/train.log b200/metrics_deepvision.csv
python3 b200/plot_progress.py
```

## 실측 근거 (2026-08-15)
- 배치는 `PDTBS 2 × ACCUM 8 × 7 = 112`. **PDTBS 4 도 돌려봤지만 되돌렸다** — `expandable_segments` 덕에
  OOM 은 안 나지만(143→150 GiB) wall clock 이 119 vs 122 s/step 로 같다. `step_time` 만 48→38 s 로 줄고
  벽시계가 그대로라는 건 병목이 step 밖(롤아웃 대기·통신)에 있다는 뜻이다. 메모리 65 GiB 를 더 쓰고 얻는 게 없다.
- `gradient_checkpointing=false` 불가 — 16K 시퀀스에서 PDTBS 1 도 OOM
- 순차 롤아웃은 GPU 절반이 절반 시간 유휴 → `--async_generate true`(server 모드 전용)로 오버랩.
  대신 롤아웃이 1 라운드 이전 가중치를 쓰므로 `--rollout_importance_sampling_mode token_truncate` 로 보정한다.
  보정이 걸렸는지는 **`ess`** 로 본다(기준선 0.9). 다만 **`ess` 만 보면 안 된다** — 08-16 실측에서
  `ess` 는 0.96 을 유지하는데 `log_ppl_abs_diff` 는 0.067→0.380 으로 벌어졌다. `ess` 는 보정 **후** 값이라
  "보정이 감당 중"만 말한다. 괴리 자체는 `log_ppl_abs_diff` 와 `training_ppl`÷`rollout_ppl` 로 봐야 한다.
- **rollout 포트 밀림**: 앞 job 이 killed 되면 그 소켓이 **TIME_WAIT** 로 남고, vLLM 은 실패하지 않고
  **조용히 8001 로 올라간다.** health 가 8000 만 보면 서버가 53초 만에 떠 있는데도 타임아웃으로 죽는다
  (교체마다 10~30분). TIME_WAIT 는 연결 체크로 감지할 수 없으므로(연결은 실패하고 bind 만 실패)
  포트를 비우려는 시도는 통하지 않는다 — `run_epoch.sh` 는 로그의 `Uvicorn running on ...:<포트>` 를
  읽어 **뜬 포트를 따라간다**(그 값이 그대로 `--vllm_server_port` 로 간다).
- 나머지 함정은 [`../CLAUDE.md`](../CLAUDE.md) §1.5 표 참조
