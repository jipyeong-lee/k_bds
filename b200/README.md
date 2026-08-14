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

진행 확인은 **job 교체 틈에서만** 가능하다(학습이 GPU 8장을 잡으면 추가 세션이 안 열린다).
그래서 `chain_epoch.sh` 가 매 job 종료 후 `check_progress.sh` 를 자동으로 한 번 돌려 로그에 남긴다.

## 실측 근거 (2026-08-15)
- 배치 상한 `PDTBS 2 × ACCUM 8 × 7 = 112` → 171 s/step. PDTBS 4 는 thrashing (`bench_pdtbs.sh` 결과)
- `gradient_checkpointing=false` 불가 — 16K 시퀀스에서 PDTBS 1 도 OOM
- 순차 롤아웃은 GPU 절반이 절반 시간 유휴 → `--async_generate true`(server 모드 전용)로 오버랩
- 나머지 함정은 [`../CLAUDE.md`](../CLAUDE.md) §1.5 표 참조
