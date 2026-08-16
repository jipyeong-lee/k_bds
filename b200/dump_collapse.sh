#!/usr/bin/env bash
# dump_collapse.sh — 붕괴 구간을 진단 지표로 뽑는다(노드에서 payload 로 실행).
#
#   FROM/TO 로 step 범위를 자른다. dump_metrics.sh 가 "곡선용 13개 열"이라면 이쪽은
#   "왜 무너졌나"를 보는 열이다 — GRPO 에서 학습이 죽는 경로는 대체로 셋이다.
#     ① reward_std 가 0 으로 수렴 → 그룹 내 advantage 소실(frac_reward_zero_std 로 확인)
#     ② grad_norm 폭발/소실
#     ③ 학습·추론 분포 괴리(training_ppl vs rollout_ppl 을 분리해서 본다)
#   stdout 은 8KB 에서 앞부분이 잘리므로 80 행 안팎으로 솎아서 내보낸다.
set -uo pipefail
: "${ORCH_HOME:?ORCH_HOME not set}"
ARM="${ARM:-deepvision}"
SCALE="${SCALE:-gdpo}"
ASYNC="${ASYNC:-true}"
ISMODE="${ISMODE:-token_truncate}"
ENTQ="${ENTQ:-0.2}"
RUNTAG="${ARM}_ep1_${SCALE}$([ "$ASYNC" = true ] && echo _async)$([ -n "$ISMODE" ] && echo _tis)$([ "$ENTQ" != "1.0" ] && echo _entmask)"
FROM="${FROM:-0}"
TO="${TO:-999999}"

"$ORCH_HOME/.venv/bin/python" - "$ORCH_HOME/train_${RUNTAG}.log" "$FROM" "$TO" <<'PY'
import re, sys
path, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cols = ['step', 'reward', 'r_std', 'zero_std', 'fmt', 'fmt_std', 'overlong',
        'len', 'len_max', 'grad', 'loss', 'tr_ppl', 'ro_ppl', 'ess', 'kl']
src = {'reward': 'reward', 'r_std': 'reward_std', 'zero_std': 'frac_reward_zero_std',
       'fmt': 'rewards/FormatThink/mean', 'fmt_std': 'rewards/FormatThink/std',
       'overlong': 'rewards/SoftOverlong/mean',
       'len': 'completions/mean_length', 'len_max': 'completions/max_length',
       'grad': 'grad_norm', 'loss': 'loss',
       'tr_ppl': 'rollout_correction/training_ppl',
       'ro_ppl': 'rollout_correction/rollout_ppl',
       'ess': 'rollout_correction/ess', 'kl': 'rollout_correction/kl'}
rows, seen = [], set()
for line in open(path, errors='ignore'):
    if 'global_step/max_steps' not in line:
        continue
    d = dict(re.findall(r"'([^']+)': '([^']*)'", line))
    g = d.get('global_step/max_steps', '')
    if '/' not in g:
        continue
    step = g.split('/')[0]
    if step in seen or not (lo <= int(step) <= hi):
        continue
    seen.add(step)
    rows.append([step] + [d.get(src[c], '') for c in cols[1:]])
LIMIT = 80
every = max(1, (len(rows) + LIMIT - 1) // LIMIT)
print(','.join(cols))
for i, r in enumerate(rows):
    if i % every == 0 or i == len(rows) - 1:
        print(','.join(r))
PY
