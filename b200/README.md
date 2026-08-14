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

## 순서
```bash
# 1) (KISTI) 업로드용 tar 생성
bash b200/make_tars.sh
# 2) (KISTI) 청크 업로드 → 플랫폼 $ORCH_HOME/uploads
ORCH_PAT=$ORCH_PAT bash b200/upload_chunked.sh /scratch/migrate_k266_to_gpu
# 3) 노드에서 셋업+스모크 (venv·데이터는 이후 세션 재사용)
ORCH_PAT=$ORCH_PAT bash b200/drive_node.sh b200/node_setup_and_smoke.sh 1 2400
```

## 3-arm 확장 (다음 단계)
`node_setup_and_smoke.sh` 4)번의 `swift rlhf`를 복제·수정:
- `--dataset $PROJ/work/data/domains/stage2_<arm>.jsonl` (deepvision/mmk12/pmcvqa)
- `--max_steps <N>` (스모크 3 → 실학습 1500)
- `--output_dir $ORCH_HOME/runs/expert_<arm>`
- GPU 배치: 노드 8장 → arm당 tensor-parallel 조정 또는 arm 순차. (`--vllm_tensor_parallel_size`, `NPROC_PER_NODE`)
- deepvision/mmk12 데이터가 아직 업로드 안 됨 → `data.tar`(46G) 청크 업로드 필요.
- 인자 근거는 KISTI `scripts/launch_domain_experts.sh` / `21_rlvr_grpo_adv.slurm` 참조(recipe=stable).
