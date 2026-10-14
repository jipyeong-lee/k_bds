#!/usr/bin/env python3
"""judge Vacuous Credit 감사 — 라이브 judge 에 medix N건 × 5변형 × (게이트 OFF/ON) 채점.

MetaRubric(arXiv 2610.02824) 의 삭제 테스트 대응: '관찰 없는 응답'이 c2·c4 를 공짜로 받는지 측정하고,
증거 인식 크레딧(RUBRIC_EVIDENCE) 프롬프트를 judge 가 지키는지(인용 실재율) 확인한다.
변형: good(근거 있는 think+정답) / vacuous_correct(관찰 없는 think+정답) / vacuous_wrong /
      wrong(근거 있는 think+오답) / halluc(엉뚱한 묘사+정답)
env: JUDGE_BASE_URL/MODEL/API_KEY, PROBE_N(기본 20), PROBE_CONC(기본 4), JUDGE_THINK=1(extra_body 생략)
실행: ./runc.sh scripts/judge_audit.py   (repo 루트에서, work/data/medix_rl_train.jsonl 필요)
"""
import os, sys, json, asyncio, statistics as st
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
import medical_reward as M  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

N = int(os.environ.get('PROBE_N', '20'))
CONC = int(os.environ.get('PROBE_CONC', '4'))
rows = []
for l in open('work/data/medix_rl_train.jsonl'):
    if len(rows) >= N:
        break
    d = json.loads(l)
    if d.get('images') and d.get('solution'):
        rows.append(d)
print(f"[audit] N={len(rows)} conc={CONC} model={os.environ['JUDGE_MODEL']}", flush=True)

cli = AsyncOpenAI(base_url=os.environ['JUDGE_BASE_URL'], api_key=os.environ.get('JUDGE_API_KEY', 'EMPTY'),
                  timeout=M.JUDGE_TIMEOUT)
sem = asyncio.Semaphore(CONC)

GROUNDED = ("On the provided image I localized the relevant region and examined its appearance "
            "(location, shape, margins, intensity) to derive the answer.")
VACUOUS = "I have read the question and will now state the final answer directly."
ABSURD = "The image clearly shows a sunny tropical beach with palm trees, blue sky and ocean waves."
WRONG = "an unrelated and incorrect answer"
VARIANTS = ['good', 'vacuous_correct', 'vacuous_wrong', 'wrong', 'halluc']


def variants(ref):
    a = ref.splitlines()[-1].strip()
    mk = lambda t, x: f"<think>{t}</think><answer>\\boxed{{{x}}}</answer>"
    return {'good': mk(GROUNDED, a), 'vacuous_correct': mk(VACUOUS, a), 'vacuous_wrong': mk(VACUOUS, WRONG),
            'wrong': mk(GROUNDED, WRONG), 'halluc': mk(ABSURD, a)}


async def score(comp, ref, img, gate):
    rubric = M.build_rubric(ref, M.is_measurement(ref))
    prompt = M.build_judge_prompt('(see image)', ref, comp, rubric, evidence=gate)
    content = [{'type': 'text', 'text': prompt}]
    du = M._image_to_data_url(img)
    if du:
        content.append({'type': 'image_url', 'image_url': {'url': du}})
    async with sem:
        try:
            r = await cli.chat.completions.create(
                model=os.environ['JUDGE_MODEL'], messages=[{'role': 'user', 'content': content}],
                temperature=0.0, max_tokens=M.JUDGE_MAX_TOKENS, extra_body=M._EXTRA_BODY)
            text = M._msg_text(r)
        except Exception as e:
            return None, f'api:{type(e).__name__}'
    v = M.parse_verdicts(text, rubric)
    if os.environ.get('AUDIT_DEBUG') and (v is None or (gate and not M.parse_evidence(text))):
        print(f"[debug gate={gate} parsed={v is not None}] {text[:500]!r}", flush=True)
    if v is None:
        return None, 'parse'
    ev = M.parse_evidence(text) if gate else ''
    raw = dict(v)
    if gate:
        v = M.apply_evidence_gate(v, ev, comp)
    return {'reward': M.aggregate(rubric, v), 'v': v, 'raw': raw, 'ev': ev}, None


async def main():
    jobs, meta = [], []
    for d in rows:
        ref = str(d['solution']).strip(); img = d['images'][0]
        for name, comp in variants(ref).items():
            for gate in (False, True):
                jobs.append(score(comp, ref, img, gate)); meta.append((name, gate, comp))
    res = await asyncio.gather(*jobs)

    for gate in (False, True):
        print(f"\n===== 게이트 {'ON ' if gate else 'OFF'} =====", flush=True)
        means = {}
        for name in VARIANTS:
            ok = [r for (r, e), (n, g, _) in zip(res, meta) if n == name and g == gate and r]
            err = [e for (r, e), (n, g, _) in zip(res, meta) if n == name and g == gate and not r]
            if not ok:
                print(f"[{name:15}] 유효 채점 0건 (실패 {len(err)}: {sorted(set(err))[:3]})"); continue
            means[name] = st.mean(r['reward'] for r in ok)
            keys = sorted({k for r in ok for k in r['v']})   # c3 는 측정형 문항에만 있다
            cr = {k: round(st.mean(r['v'][k] for r in ok if k in r['v']), 2) for k in keys}
            print(f"[{name:15}] reward={means[name]:.3f} n={len(ok)} 실패={len(err)} criterion={cr}")
        if 'good' in means and means['good'] > 0 and 'vacuous_correct' in means:
            print(f"  → 유지율 vacuous_correct/good = {means['vacuous_correct']/means['good']*100:.0f}%"
                  f"  (논문 삭제 테스트: 필요내용 삭제 후 70~82% 유지가 문제)")

    if True:
        ok = [(r, c) for (r, e), (n, g, c) in zip(res, meta) if n == 'good' and g and r]
        if ok:
            kept = sum(1 for r, c in ok if r['v']['c2'] == 1.0)
            has = sum(1 for r, c in ok if r['ev'].strip())
            print(f"\n[인용 준수] good 변형 {len(ok)}건 중 인용 비어있지 않음 {has}, 게이트 통과(c2 유지) {kept}")
            print("   예시 인용:", [r['ev'][:60] for r, c in ok[:2]])


asyncio.run(main())
