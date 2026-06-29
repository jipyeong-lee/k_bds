#!/bin/bash
# =============================================================================
# 32_net_test.slurm — 컴퓨트노드 간 내부망 도달성 테스트 (Stage-3 judge 전제 확인)
#   2노드 할당: node0 = HTTP 리스너, node1 = 접속 시도(hostname·IP). judge 모델 불필요(빠름).
#   ✅ 통과 = 학습노드↔judge노드 내부 통신 가능 → judge self-host 경로 확정.
# =============================================================================
#SBATCH --job-name=net-test
#SBATCH --partition=cpu32-only
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:08:00
#SBATCH --comment=etc
#SBATCH --output=/home01/k252a01/kbds_project/logs/net_test_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/net_test_%j.log

set -u
cd /home01/k252a01/kbds_project
PY=/home01/k252a01/.conda/envs/swift/bin/python3   # 공유(/home01) 경로 → 전 노드 가시
PORT="${NET_TEST_PORT:-8137}"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SERVER=${NODES[0]}; CLIENT=${NODES[1]}
echo "[net-test] server=$SERVER  client=$CLIENT  port=$PORT"

# server IP (배치 노드에서 DNS 해석 — srun 안 씀, 슬롯 블록 방지)
SERVER_IP=$(getent hosts "$SERVER" | awk '{print $1}' | head -1)
echo "[net-test] server hostname=$SERVER  resolved IP=$SERVER_IP"

# 1) node0: HTTP 리스너 백그라운드 (0.0.0.0 바인드, --overlap 로 슬롯 경합 회피)
srun --overlap --nodes=1 --ntasks=1 --nodelist="$SERVER" bash -c "
  echo '[server] host=\$(hostname)'; hostname -I
  $PY -m http.server $PORT --bind 0.0.0.0
" &
SRV=$!
sleep 10

# 2) node1 → node0 접속 시도 (hostname + IP)
srun --overlap --nodes=1 --ntasks=1 --nodelist="$CLIENT" bash -c "
  echo '[client] from \$(hostname) → $SERVER ($SERVER_IP):$PORT'
  echo -n '  hostname 접속: '; (curl -s --max-time 10 -o /dev/null -w '%{http_code}' http://$SERVER:$PORT/ && echo ' ✅') || echo ' ❌'
  echo -n '  IP 접속:       '; (curl -s --max-time 10 -o /dev/null -w '%{http_code}' http://$SERVER_IP:$PORT/ && echo ' ✅') || echo ' ❌'
  echo '  (200/404 등 코드 = TCP 도달 성공 / 무응답·타임아웃 = 차단)'
"
RC=$?

# 4) teardown
kill $SRV 2>/dev/null; sleep 1; kill -9 $SRV 2>/dev/null || true
echo "[net-test] done (rc=$RC)"
