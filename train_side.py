#!/usr/bin/env python3
"""학습측 형식보상·길이 곡선을 홀드아웃 지점에 맞춰 집계.

v0/v1 두 run 을 합친다. resume 로 겹치는 구간(751~787)은 v1 을 채택 —
v1 이 실제로 이어서 학습한 쪽이다(v0 replay 는 §8-e 에서 near-identical 확인).
"""
import json, glob, statistics as st, pathlib

#  구 계정 절대경로가 박혀 있었다(k252a02) → 이관 후 파일을 못 연다. __file__ 기준으로 바꾼다.
_CK = pathlib.Path(__file__).parent / 'work/checkpoints/grpo_expanded_gdpo'
V = [str(_CK / 'v0-20260731-094532/logging.jsonl'),
     str(_CK / 'v1-20260803-074645/logging.jsonl')]

rows = {}
for i, p in enumerate(V):
    for line in open(p):
        try:
            d = json.loads(line)
        except Exception:
            continue
        gs = d.get('global_step/max_steps')
        if not gs:
            continue
        s = int(str(gs).split('/')[0])
        if i == 1 or s not in rows:      # v1 이 이김
            rows[s] = d

steps = sorted(rows)
print(f'step 범위 {steps[0]}~{steps[-1]}  (n={len(steps)})')

K = [('rewards/FormatThink/mean', 'FormatThink'),
     ('rewards/AccuracyMix/mean', 'AccuracyMix'),
     ('rewards/SoftOverlong/mean', 'SoftOverlong'),
     ('completions/mean_length', 'mean_len(tok)'),
     ('completions/clipped_ratio', 'clipped'),
     ('entropy/mean', 'entropy'),
     ('kl', 'KL'),
     ('frac_reward_zero_std', 'zero_std')]

MARKS = [50, 400, 500, 600, 700, 800, 850, 900, 950, 1000, 1047]
W = 50


def win(lo, hi, key):
    v = [rows[s][key] for s in steps if lo < s <= hi and key in rows[s] and rows[s][key] is not None]
    return st.mean(v) if v else None


print('\n' + '=' * 108)
print(f'학습측 지표 — 각 지점 직전 {W} step 평균')
print('=' * 108)
hdr = f"{'~step':>7}" + ''.join(f'{lbl:>15}' for _, lbl in K)
print(hdr)
for m in MARKS:
    line = f'{m:>7}'
    for k, _ in K:
        v = win(m - W, m, k)
        line += ('     —' .rjust(15)) if v is None else (f'{v:>15.4f}' if abs(v) < 100 else f'{v:>15.0f}')
    print(line)

# 붕괴 구간 세밀
print('\n' + '=' * 108)
print('붕괴 구간 10 step 평균')
print('=' * 108)
print(hdr)
for m in range(780, 1051, 10):
    line = f'{m:>7}'
    for k, _ in K:
        v = win(m - 10, m, k)
        line += ('     —'.rjust(15)) if v is None else (f'{v:>15.4f}' if abs(v) < 100 else f'{v:>15.0f}')
    print(line)
