#!/usr/bin/env python3
"""eval_paired.py — 두 체크포인트의 홀드아웃 결과를 문항별로 짝지어 McNemar 검정.

왜 필요한가: 같은 문항을 두 모델이 푼 것이므로 짝지음 검정이 맞다. 집계값만 비교하면
비짝지음 오차(n=300 에서 검출 하한 8.0pp)를 쓰게 되어, 실제로는 존재하는 3pp 수준의
차이를 "차이 없음"으로 오독한다. 짝지음이면 같은 n 에서 하한이 5.7pp, n=1,200 이면 2.8pp 다.

입력은 eval_compare.py 가 EVAL_ITEMS 로 떨군 문항별 jsonl 두 개.
표준 라이브러리만 사용 — 호스트 python3 로 그냥 돈다.

  python3 scripts/eval_paired.py logs/eval_items_step400_*.jsonl logs/eval_items_step1200_*.jsonl
  python3 scripts/eval_paired.py A.jsonl B.jsonl --by source     # 소스별로도 쪼개서
"""
import argparse
import json
import math
from collections import defaultdict


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r['item_id']] = r
    return rows


def mcnemar(pairs):
    """pairs: [(a_score, b_score)] → (delta_pp, b_only, c_only, p, mde_pp)

    b = A 만 맞음, c = B 만 맞음. 정확검정(이항, 양측)을 쓴다 — b+c 가 작을 때
    카이제곱 근사는 p 를 과소평가한다.
    """
    n = len(pairs)
    b = sum(1 for a, x in pairs if a >= 0.5 > x)      # A만 정답
    c = sum(1 for a, x in pairs if x >= 0.5 > a)      # B만 정답
    delta = (c - b) / n * 100 if n else 0.0
    d = b + c
    if d == 0:
        p = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(d, i) for i in range(k + 1)) / (2 ** d)
        p = min(1.0, 2 * tail)
    mde = 1.96 * math.sqrt(d) / n * 100 if n else float('nan')
    return delta, b, c, p, mde


def report(title, pairs):
    n = len(pairs)
    if not n:
        print(f"  {title:<12} (짝지어진 문항 없음)")
        return
    accA = sum(a for a, _ in pairs) / n
    accB = sum(x for _, x in pairs) / n
    delta, b, c, p, mde = mcnemar(pairs)
    verdict = "유의" if p < 0.05 else "미검출"
    flag = "  ← 검출 하한 미만" if abs(delta) < mde else ""
    print(f"  {title:<12} n={n:<5} A={accA:.4f}  B={accB:.4f}  "
          f"Δ={delta:+.2f}pp  (A만 {b} / B만 {c})  p={p:.4f} {verdict}"
          f"  |  이 표본의 검출 하한 ±{mde:.2f}pp{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a', help='기준 체크포인트의 문항별 jsonl (예: step400)')
    ap.add_argument('b', help='비교 체크포인트의 문항별 jsonl (예: step1200)')
    ap.add_argument('--by', default='source', choices=('source', 'stratum', 'none'),
                    help='전체 외에 추가로 쪼개 볼 축 (기본 source)')
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    tagA = next(iter(A.values()))['tag']
    tagB = next(iter(B.values()))['tag']

    print(f"A = {tagA}  ({args.a}, {len(A)} 문항)")
    print(f"B = {tagB}  ({args.b}, {len(B)} 문항)")
    print(f"짝지어진 문항: {len(common)}"
          + ("" if len(common) == len(A) == len(B)
             else "   ⚠️ 슬라이스가 다르다 — EVAL_N·EVAL_SEED 가 같은지 확인할 것"))
    print()

    pairs = [(A[k]['score'], B[k]['score']) for k in common]
    report('전체', pairs)

    if args.by != 'none':
        groups = defaultdict(list)
        for k in common:
            groups[A[k].get(args.by, 'all')].append((A[k]['score'], B[k]['score']))
        if len(groups) > 1:
            print()
            for g in sorted(groups):
                report(g, groups[g])

    print("\n※ 'Δ 가 검출 하한 미만' 은 '차이 없음' 이 아니라 '이 표본으로는 볼 수 없음' 이다.")


if __name__ == '__main__':
    main()
