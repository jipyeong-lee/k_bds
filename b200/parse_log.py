#!/usr/bin/env python3
"""학습 로그(train_<RUNTAG>.log)를 통째로 받아 CSV 로 뽑는다.

    bash b200/pull_file.sh train_deepvision_ep1_gdpo_async_tis.log /tmp/train.log
    python3 b200/parse_log.py /tmp/train.log b200/metrics_deepvision.csv

`dump_metrics.sh` 는 노드 안에서 돌기 때문에 (a) GPU 세션이 비는 틈을 기다려야 하고
(b) stdout 8KB 제한 때문에 80 행으로 솎아야 했다. `/me/data/file` 로 로그를 통째로
내려받을 수 있으니 두 제약이 다 없어진다 — 여기서는 전 step 을 그대로 남긴다.
"""
import csv
import re
import sys

COLS = ['step', 'reward', 'acc', 'fmt', 'len', 'clipped', 'mem', 'step_time', 'lr',
        'ppl_abs_diff', 'ess', 'is_weight', 'clipped_frac',
        'r_std', 'zero_std', 'grad', 'tr_ppl', 'ro_ppl']
SRC = {
    'reward': 'reward', 'acc': 'rewards/AccuracyMix/mean', 'fmt': 'rewards/FormatThink/mean',
    'len': 'completions/mean_length', 'clipped': 'completions/clipped_ratio',
    'mem': 'memory(GiB)', 'step_time': 'step_time', 'lr': 'learning_rate',
    'ppl_abs_diff': 'rollout_correction/log_ppl_abs_diff',
    'ess': 'rollout_correction/ess',
    'is_weight': 'rollout_correction/is_weight_mean',
    'clipped_frac': 'rollout_correction/clipped_frac',
    'r_std': 'reward_std', 'zero_std': 'frac_reward_zero_std', 'grad': 'grad_norm',
    'tr_ppl': 'rollout_correction/training_ppl',
    'ro_ppl': 'rollout_correction/rollout_ppl',
}

src = sys.argv[1] if len(sys.argv) > 1 else '/tmp/train.log'
dst = sys.argv[2] if len(sys.argv) > 2 else 'b200/metrics_deepvision.csv'

rows, seen = [], set()
for line in open(src, errors='ignore'):
    if 'global_step/max_steps' not in line:
        continue
    d = dict(re.findall(r"'([^']+)': '([^']*)'", line))
    g = d.get('global_step/max_steps', '')
    if '/' not in g:
        continue
    step = g.split('/')[0]
    if step in seen:      # 재개하면 같은 step 이 다시 찍힌다 → 처음 것만.
        continue
    seen.add(step)
    rows.append([step] + [d.get(SRC[c], '') for c in COLS[1:]])

rows.sort(key=lambda r: int(r[0]))
with open(dst, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(COLS)
    w.writerows(rows)
print(f"{dst}: {len(rows)} steps ({rows[0][0]}~{rows[-1][0]})")
