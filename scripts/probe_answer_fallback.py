#!/usr/bin/env python3
"""probe_answer_fallback.py — `<answer>` 태그 소실 시 정확도 보상 폴백의 실측 설계·검증.

배경(사후분석 §4-4): step 899 붕괴의 되먹임 고리는 **정확도 보상이 형식에 결합**돼 있어서다.
`configs/accuracy.py:_strip_answer` 는 `<answer>` 태그가 없으면 **추론 전문(全文)** 을 답안으로
넘긴다. letter 경로는 `_LETTER_PICK` 이 `^` 앵커라 구조적으로 0 점이 되고, 형식을 잃는 순간
가중치 1.0 의 정확도까지 0 이 된다 → advantage 가 "정답이냐"가 아니라 "형식 지켰냐"로 지배됨.

⚠️ 실측으로 확인한 붕괴 출력의 실제 모양(추측이 아니다):
      "...This corresponds to option B.\\n</think><|im_end|>"
   **`</think>` 뒤가 비어 있다.** 모델이 추론을 끝내고 답을 안 쓰고 멈춘다.
   따라서 폴백은 `</think>` **뒤**가 아니라 추론 **꼬리 자체**를 봐야 한다.
   (초판은 `</think>` 뒤를 훑도록 짰다가 전량 놓쳤다 — 그래서 이 스크립트가 있다)

   꼬리 300자에 A~H 대문자가 있는 비율은 **52%** 다. 나머지는 모델이 letter 를 말하지도
   않고 멈춘 것이라 **원리적으로 회수 불가**다. 이게 회수율의 천장이다.

두 축으로 잰다.
  ① 정밀도 — 태그가 **있는** 롤아웃에서 태그만 지우고 폴백을 돌려, 태그 경로 점수를
     복원하는지 본다. 거짓양성이 높으면 형식을 버려도 점수가 나온다 = 보상 해킹.
  ② 회수율 — 붕괴 구간 태그 **없는** 롤아웃에서 현행 대비 얼마나 되살아나는가.

letter 추출기와 math 추출기를 **따로** 평가한다(둘의 최적이 다르다).

사용:
    ./bin/python scripts/probe_answer_fallback.py \
        work/checkpoints/grpo_expanded_gdpo/v1-20260803-074645/completions.jsonl
  * math_verify 필요 → 반드시 `./bin/python`(loader) 으로 실행.
"""
import argparse
import json
import re
import sys
from collections import Counter

# ── 현행 configs/accuracy.py 에서 그대로 옮긴 것 (swift 의존 제거) ────────────────
_ANS_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
_BOXED_RE = re.compile(r'\\boxed\{(.*?)\}', re.DOTALL)
_LETTER_RE = re.compile(r'^[A-Ha-h]$')
_LETTER_PICK = re.compile(r'^\(?([A-Ha-h])\)?[\s.):]', re.DOTALL)


def _strip_answer(text):
    m = _ANS_RE.search(text)
    return (m.group(1).strip() if m else text.strip())


def _norm_pred(ans):
    b = _BOXED_RE.search(ans)
    if b:
        ans = b.group(1).strip()
    ans = ans.strip().strip('()[]{}').strip()
    return ans.rstrip('.').strip()


def _letter_match(pred_ans, gold_letter):
    p = _norm_pred(pred_ans)
    if _LETTER_RE.match(p):
        return float(p.upper() == gold_letter.upper())
    m = _LETTER_PICK.match(p)
    if m:
        return float(m.group(1).upper() == gold_letter.upper())
    return 0.0


# ── 폴백 후보 ────────────────────────────────────────────────────────────────
_TAIL = 400                       # 훑을 꼬리 길이(문자)
_SPECIAL = re.compile(r'<\|im_end\|>|<\|endoftext\|>|</?think>|</?answer>')

# 데이터에서 실측한 답 신호(빈도순): "the correct answer is X" · "corresponds to option X"
# · "the correct choice is X" · "the correct option is X" · "matches option X"
_CUE_FWD = re.compile(
    r'(?:answer|option|choice|정답|답)\s*(?:is|:|=|→|->|은|는)?\s*'
    r'[\*\'"“”\s(\[]*([A-H])(?![A-Za-z])', re.IGNORECASE)
# 역방향: "'B.(0, 0)' corresponds to this coordinate" — letter 가 신호 **앞**에 온다
_CUE_BWD = re.compile(
    r'(?<![A-Za-z])([A-H])[\)\.\'"”\s]*(?:is\s+the\s+)?'
    r'(?:corresponds|matches|is\s+correct|is\s+the\s+answer)', re.IGNORECASE)
# 최후 수단: 꼬리 끝에 홀로 선 **대문자** A~H. 소문자 a 는 영어 관사라 절대 받지 않는다.
_BARE_UPPER = re.compile(r'(?<![A-Za-z])\(?([A-H])\)?[\s.):,\'"]*$')
_BOXED_ANY = re.compile(r'\\boxed\{([^}]*)\}')
_NUM = re.compile(r'-?\d+(?:\.\d+)?')


def _tail_of(text):
    """결론부. ⚠️ `</think>` 뒤를 자르지 않는다 — 붕괴 출력은 그 뒤가 비어 있다."""
    t = _SPECIAL.sub(' ', text).strip()
    return t[-_TAIL:]


def extract_letter(text, mode):
    """태그 없는 출력에서 선택지 letter 추출. None = 추출 실패(= 0점)."""
    if mode == 'current':
        return None
    t = _tail_of(text)
    for cand in reversed(_BOXED_ANY.findall(t)):          # \boxed{B} 는 강한 신호
        c = cand.strip().strip('()[]{}').strip()
        if _LETTER_RE.match(c):
            return c.upper()
    hits = [(m.end(), m.group(1)) for m in _CUE_FWD.finditer(t)]
    hits += [(m.end(), m.group(1)) for m in _CUE_BWD.finditer(t)]
    if hits:
        return max(hits)[1].upper()                       # 가장 뒤의 신호를 쓴다
    if mode == 'cue':
        return None
    m = _BARE_UPPER.search(t)                             # mode == 'cue_bare'
    return m.group(1).upper() if m else None


def extract_math(text, mode, parse):
    """태그 없는 출력에서 최종 수치 추출. 'current' 는 전문 first_match(현행)."""
    if mode == 'current':
        return text.strip(), 'first_match'
    if mode == 'last_full':
        return text.strip(), 'last_match'
    t = _tail_of(text)                                    # mode == 'tail'
    b = _BOXED_ANY.findall(t)
    if b:
        return b[-1].strip(), 'first_match'
    cue = re.search(r'(?:answer|정답|답)\s*(?:is|:|=|→|->|은|는)?\s*'
                    r'[\*\s]*(-?\d+(?:\.\d+)?)', t, re.IGNORECASE)
    if cue:
        return cue.group(1), 'first_match'
    nums = _NUM.findall(t)
    return (nums[-1], 'first_match') if nums else (None, None)


# ── 채점 ────────────────────────────────────────────────────────────────────
def gold_kind(gold, parse):
    if _LETTER_RE.match(gold):
        return 'letter'
    try:
        if len(parse(gold, extraction_mode='first_match')) != 0:
            return 'math'
    except Exception:
        pass
    return 'string'


def score(text, gold, kind, lmode, mmode, parse, verify):
    m = _ANS_RE.search(text)
    tagged = m is not None
    ans = m.group(1).strip() if tagged else text.strip()

    if kind == 'math':
        if tagged:
            src, em = ans, 'first_match'
        else:
            src, em = extract_math(text, mmode, parse)
        if src is None:
            return 0.0
        try:
            gp = parse(gold, extraction_mode='first_match')
            return float(verify(gp, parse(src, extraction_mode=em)))
        except Exception:
            return 0.0

    if kind == 'letter':
        if tagged:
            return _letter_match(ans, gold)
        got = extract_letter(text, lmode)
        return 0.0 if got is None else float(got == gold.upper())

    if tagged or lmode == 'current':
        return float(_norm_pred(ans).casefold() == gold.casefold())
    return float(_norm_pred(_tail_of(text)).casefold() == gold.casefold())


LETTER_MODES = ['current', 'cue', 'cue_bare']
MATH_MODES = ['current', 'last_full', 'tail']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('completions')
    ap.add_argument('--collapse-from', type=int, default=899)
    a = ap.parse_args()
    from math_verify import parse, verify

    rows = []
    for line in open(a.completions):
        try:
            r = json.loads(line)
        except Exception:
            continue
        for i in range(len(r.get('completion', []))):
            rows.append({'step': int(r['step'][i]), 'completion': r['completion'][i],
                         'solution': str(r['solution'][i]), 'acc': float(r['AccuracyMix'][i]),
                         'fmt': float(r['FormatThink'][i])})
    for x in rows:
        x['gold'] = _strip_answer(x['solution'])
        x['kind'] = gold_kind(x['gold'], parse)
        x['tagged'] = _ANS_RE.search(x['completion']) is not None
    print(f"[load] {len(rows):,} 롤아웃 · step {min(x['step'] for x in rows)}"
          f"~{max(x['step'] for x in rows)}")

    bad = sum(abs(score(x['completion'], x['gold'], x['kind'], 'current', 'current',
                        parse, verify) - x['acc']) > 1e-6 for x in rows)
    print(f"[검증] 현행 로직 재현: 불일치 {bad:,}/{len(rows):,} ({100*bad/len(rows):.2f}%)"
          f"{'  ✅' if bad/len(rows) < 0.02 else '  🚨 재현 실패 — 아래 수치 신뢰 불가'}")
    print(f"[분포] gold 유형: " +
          " · ".join(f"{k} {v:,}" for k, v in Counter(x['kind'] for x in rows).most_common()))
    print(f"[분포] 태그 있음 {sum(x['tagged'] for x in rows):,} · "
          f"없음 {sum(not x['tagged'] for x in rows):,}")

    tagged = [x for x in rows if x['tagged']]
    lost = [x for x in rows if x['step'] >= a.collapse_from and not x['tagged']]

    def report(kind, modes, kw):
        print(f"\n{'='*72}\n[{kind}] 추출기 비교\n{'='*72}")
        tg = [x for x in tagged if x['kind'] == kind]
        ls = [x for x in lost if x['kind'] == kind]
        print(f"{'모드':<11} | {'① 정밀도(태그 제거·n=' + f'{len(tg):,}' + ')':^30} | "
              f"{'② 회수(붕괴·n=' + f'{len(ls):,}' + ')':^24}")
        print(f"{'':<11} | {'일치':>8} {'거짓양성':>9} {'놓침':>9} | {'현행':>7} {'폴백':>7} {'Δ':>7}")
        print('-'*74)
        cur = (sum(x['acc'] for x in ls) / len(ls)) if ls else 0.0
        for mode in modes:
            kw2 = dict(kw); kw2[kind == 'letter' and 'lmode' or 'mmode'] = mode
            ag = fp = fn = 0
            for x in tg:
                st = _ANS_RE.sub(lambda m: m.group(1), x['completion'])
                truth = score(x['completion'], x['gold'], kind, 'current', 'current',
                              parse, verify)
                got = score(st, x['gold'], kind, kw2['lmode'], kw2['mmode'], parse, verify)
                ag += got == truth; fp += got > truth; fn += got < truth
            new = (sum(score(x['completion'], x['gold'], kind, kw2['lmode'], kw2['mmode'],
                             parse, verify) for x in ls) / len(ls)) if ls else 0.0
            n = max(len(tg), 1)
            print(f"{mode:<11} | {100*ag/n:>7.1f}% {100*fp/n:>8.2f}% {100*fn/n:>8.1f}% | "
                  f"{cur:>7.4f} {new:>7.4f} {new-cur:>+7.4f}")

    report('letter', LETTER_MODES, {'lmode': 'current', 'mmode': 'current'})
    report('math', MATH_MODES, {'lmode': 'current', 'mmode': 'current'})

    # ── 절벽: 형식 유무에 따른 총보상 격차 ───────────────────────────────────
    print(f"\n{'='*72}\n절벽 — 형식을 잃을 때 총보상 낙폭 (가중치 acc 1.0 · fmt 0.2)\n{'='*72}")
    ok = [x for x in rows if x['step'] < a.collapse_from and x['fmt'] == 1.0]
    acc_ok = sum(x['acc'] for x in ok) / len(ok)
    base = acc_ok + 0.2
    print(f"붕괴 전 형식 정상 {len(ok):,} · 평균 정확도 {acc_ok:.4f} → 총보상 {base:.4f}")
    print(f"\n{'시나리오':<34} {'정확도':>8} {'총보상':>8} {'낙폭':>9}")
    print('-'*62)
    combos = [('현행 (letter=current, math=current)', 'current', 'current')]
    combos += [(f'letter={lm}, math={mm}', lm, mm)
               for lm in ['cue', 'cue_bare'] for mm in ['current', 'last_full', 'tail']]
    for lab, lm, mm in combos:
        new = sum(score(x['completion'], x['gold'], x['kind'], lm, mm, parse, verify)
                  for x in lost) / len(lost)
        print(f"{lab:<34} {new:>8.4f} {new:>8.4f} {new-base:>+9.4f}")
    print("\n낙폭이 작을수록 되먹임 고리가 약하다. 0 에 가까우면 형식을 버려도 손해가 없어지므로"
          "\n형식 가중치로 **최소 낙폭**은 남겨야 한다 — 목표는 제거가 아니라 절벽→경사다.")


if __name__ == '__main__':
    sys.exit(main())
