#!/usr/bin/env bash
# dump_metrics.sh — 학습 로그를 CSV 로 뽑는다(노드에서 payload 로 실행).
# ms-swift 는 매 step 한 줄에 dict 를 통째로 찍는다(수천 자). 그대로 가져오면 stdout 을 잡아먹으므로
# 필요한 열만 골라 CSV 로 줄인다. RUNTAG 는 run_epoch.sh 와 같은 규칙으로 만들어야 로그를 찾는다.
set -uo pipefail
: "${ORCH_HOME:?ORCH_HOME not set}"
ARM="${ARM:-deepvision}"
SCALE="${SCALE:-gdpo}"
ASYNC="${ASYNC:-true}"
ISMODE="${ISMODE:-token_truncate}"
RUNTAG="${ARM}_ep1_${SCALE}$([ "$ASYNC" = true ] && echo _async)$([ -n "$ISMODE" ] && echo _tis)"

"$ORCH_HOME/.venv/bin/python" - "$ORCH_HOME/train_${RUNTAG}.log" "${1:-1}" <<'PY'
import re, sys
path, every = sys.argv[1], int(sys.argv[2])
cols = ['step','reward','acc','fmt','len','clipped','mem','step_time','lr',
        'ppl_abs_diff','ess','is_weight','clipped_frac']
src = {'reward':'reward', 'acc':'rewards/AccuracyMix/mean', 'fmt':'rewards/FormatThink/mean',
       'len':'completions/mean_length', 'clipped':'completions/clipped_ratio',
       'mem':'memory(GiB)', 'step_time':'step_time', 'lr':'learning_rate',
       'ppl_abs_diff':'rollout_correction/log_ppl_abs_diff',
       'ess':'rollout_correction/ess',
       'is_weight':'rollout_correction/is_weight_mean',
       'clipped_frac':'rollout_correction/clipped_frac'}
rows, seen = [], set()
for line in open(path, errors='ignore'):
    if 'global_step/max_steps' not in line:
        continue
    d = dict(re.findall(r"'([^']+)': '([^']*)'", line))
    g = d.get('global_step/max_steps', '')
    if '/' not in g:
        continue
    step = g.split('/')[0]
    # 재개하면 같은 step 이 다시 찍힐 수 있다 → 처음 것만 남긴다.
    if step in seen:
        continue
    seen.add(step)
    rows.append([step] + [d.get(src[c], '') for c in cols[1:]])

# run_node_extract 의 stdout 은 8KB 근처에서 잘린다 — 전부 내보내면 앞부분이 통째로 사라지고
# 뒷부분만 남는다(실제로 289 step 중 191~283 만 돌아왔다). 그래서 여기서 미리 솎아낸다.
# 마지막 행은 현재 상태라 반드시 남긴다.
LIMIT = 80
every = max(1, (len(rows) + LIMIT - 1) // LIMIT) if every <= 1 else every
print(','.join(cols))
for i, r in enumerate(rows):
    if i % every == 0 or i == len(rows) - 1:
        print(','.join(r))
PY
