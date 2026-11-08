#!/usr/bin/env python3
"""overthink_probe.py — 기존 rollout 에서 과잉 사고 신호를 잰다(읽기 전용, LoCo 적용 전 사전 점검).
사용: python3 scripts/overthink_probe.py [rubric_rollouts.jsonl]
지표: <think> 길이 분포 · 확인 표현 빈도 · 같은 문제 8개 중 정답(reward 최대) 가운데 가장 짧은 것 대비 낭비율."""
import json, re, statistics as st, sys
P = sys.argv[1] if len(sys.argv) > 1 else 'work/rubric_ablation/rubric_rollouts.jsonl'
S = 'work/rubric_ablation/scores.jsonl'
CHK = re.compile(r'\b(wait|let me (double|re)?-?check|let me verify|double-check|hmm|alternatively|on second thought)\b', re.I)
rew = {}
for l in open(S):
    r = json.loads(l)
    if r['cfg'] == 'C0': rew[r['key']] = (r.get('reward') or 0.0, r.get('tokens'))
L, chk, waste, n_think = [], [], [], 0
for l in open(P):
    r = json.loads(l); cs = json.loads(r['completions']) if isinstance(r['completions'], str) else r['completions']
    lens = [len(c.split()) for c in cs]; L += lens
    for c in cs:
        m = re.search(r'<think>(.*?)</think>', c, re.S)
        if m: n_think += 1; chk.append(len(CHK.findall(m.group(1))))
    good = [lens[i] for i in range(len(cs)) if rew.get(f"{r['id']}#{i}", (0, 0))[0] >= 0.99]
    if len(good) >= 2: waste.append(1 - min(good) / st.mean(good))
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print(f'rollout {len(L)}개 · think 블록 {n_think}개 · 단어수 중앙 {q(L,.5)} p90 {q(L,.9)} 최대 {max(L)}')
print(f'think 안 확인 표현: 평균 {st.mean(chk):.1f}회 · 5회 이상 {sum(c>=5 for c in chk)/len(chk):.0%}' if chk else 'think 블록 없음')
print(f'정답 2개 이상인 문제 {len(waste)}개 · 평균 길이가 최단 정답보다 길어 낭비되는 몫 평균 {st.mean(waste):.0%}' if waste else '비교 가능한 문제 없음')
