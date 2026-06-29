#!/usr/bin/env python3
"""정적 통일 루브릭 vs 인스턴스(논문식 RaR) 루브릭 — 동일 medix 샘플에 둘 다 적용해 비교.

- 정적: medical_reward.build_rubric (4차원 통일 + 참조답 주입)
- 인스턴스: judge LLM 이 질문+참조답으로 루브릭을 생성(논문 방식, 이미지 없이 텍스트 기반) → 그걸로 채점
각 샘플 good/wrong/halluc 3변형 채점 → 단조성·halluc 변별·항목수 비교.
env: JUDGE_BASE_URL/MODEL, CMP_N(40), CMP_CONC(8).
"""
import os, sys, re, json, asyncio, statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
import medical_reward as M  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

N = int(os.environ.get('CMP_N', '40'))
CONC = int(os.environ.get('CMP_CONC', '8'))
rows = []
for l in open('work/data/medix_rl_train.jsonl'):
    if len(rows) >= N:
        break
    d = json.loads(l)
    if d.get('images') and d.get('solution'):
        rows.append(d)
print(f"[cmp] N={len(rows)} conc={CONC} model={os.environ['JUDGE_MODEL']}")

cli = AsyncOpenAI(base_url=os.environ['JUDGE_BASE_URL'], api_key='EMPTY')
MODEL = os.environ['JUDGE_MODEL']
sem = asyncio.Semaphore(CONC)

GROUNDED = "On the image I localized the relevant region and examined its appearance to derive the answer."
ABSURD = "The image clearly shows a sunny tropical beach with palm trees, blue sky and ocean waves."
_JSONLIST = re.compile(r'\[.*\]', re.S)
GEN_SYS = (
    "You design a grading rubric for a medical image VQA question. Given the QUESTION and the REFERENCE "
    "ANSWER (gold), output a concise checklist of 2-5 binary criteria that a correct answer must satisfy "
    "(Rubrics-as-Rewards style). Output ONLY a JSON list; each item is "
    '{"title": str, "description": str, "weight": int} where description begins with '
    "'Essential Criteria:' (weight 5) / 'Important Criteria:' (weight 3) / 'Pitfall Criteria:' (weight 4, "
    "positively phrased). You do NOT see the image; base the criteria on the reference answer.")


def variants(ref):
    ans = ref.splitlines()[-1].strip()
    return {'good':   f"<think>{GROUNDED}</think><answer>\\boxed{{{ans}}}</answer>",
            'wrong':  f"<think>{GROUNDED}</think><answer>\\boxed{{an unrelated and incorrect answer}}</answer>",
            'halluc': f"<think>{ABSURD}</think><answer>\\boxed{{{ans}}}</answer>"}


async def gen_rubric(q, ref):
    msg = [{'role': 'user', 'content': f"{GEN_SYS}\n\nQUESTION: {q}\nREFERENCE ANSWER: {ref}"}]
    async with sem:
        try:
            r = await cli.chat.completions.create(model=MODEL, messages=msg, temperature=0.0,
                                                  max_tokens=512, extra_body=M._EXTRA_BODY)
            m = _JSONLIST.search(M._msg_text(r)); arr = json.loads(m.group(0))
        except Exception:
            return None
    rub = []
    for i, it in enumerate(arr[:6]):
        if isinstance(it, dict) and it.get('description'):
            try:
                w = int(it.get('weight', 3))
            except Exception:
                w = 3
            rub.append({'key': f'c{i+1}', 'title': str(it.get('title', f'c{i+1}')),
                        'weight': w, 'description': str(it['description'])})
    return rub or None


async def score(rubric, comp, ref, img):
    if not M.format_ok(comp):
        return 0.0
    content = [{'type': 'text', 'text': M.build_judge_prompt('(see image)', ref, comp, rubric)}]
    du = M._image_to_data_url(img)
    if du:
        content.append({'type': 'image_url', 'image_url': {'url': du}})
    async with sem:
        try:
            r = await cli.chat.completions.create(model=MODEL, messages=[{'role': 'user', 'content': content}],
                                                  temperature=0.0, max_tokens=M.JUDGE_MAX_TOKENS, extra_body=M._EXTRA_BODY)
            v = M.parse_verdicts(M._msg_text(r), rubric)
        except Exception:
            return 0.0
    return M.aggregate(rubric, v) if v else 0.0


async def one(d):
    ref = str(d['solution']).strip(); img = d['images'][0]
    q = d['messages'][0]['content'].replace('<image>', '').strip()
    var = variants(ref)
    static_rb = M.build_rubric(ref, M.is_measurement(ref))
    inst_rb = await gen_rubric(q, ref)
    out = {'n_static': len(static_rb), 'n_inst': (len(inst_rb) if inst_rb else 0)}
    for name, comp in var.items():
        out[f'static_{name}'] = await score(static_rb, comp, ref, img)
        out[f'inst_{name}'] = (await score(inst_rb, comp, ref, img)) if inst_rb else None
    return out


async def main():
    res = await asyncio.gather(*[one(d) for d in rows])

    def m(prefix, name):
        v = [r[f'{prefix}_{name}'] for r in res if r.get(f'{prefix}_{name}') is not None]
        return st.mean(v) if v else float('nan')

    print(f"\n{'='*54}")
    print("  정적 통일 루브릭  vs  인스턴스(논문식) 루브릭")
    print(f"{'='*54}")
    print(f"평균 항목수: 정적 {st.mean(r['n_static'] for r in res):.1f} / 인스턴스 {st.mean(r['n_inst'] for r in res):.1f}")
    print(f"인스턴스 생성 실패: {sum(1 for r in res if r['n_inst']==0)}/{len(res)}")
    for prefix, lbl in [('static', '정적 통일'), ('inst', '인스턴스')]:
        g, w, h = m(prefix, 'good'), m(prefix, 'wrong'), m(prefix, 'halluc')
        mono = sum(1 for r in res if (r.get(f'{prefix}_good') or 0) > (r.get(f'{prefix}_wrong') or 1)) / len(res)
        print(f"\n[{lbl:8}] good={g:.3f} wrong={w:.3f} halluc={h:.3f}")
        print(f"            단조성(good>wrong)={mono*100:.0f}%  |  halluc 변별(good−halluc)={g-h:+.3f}")
    print("\n(halluc 변별이 클수록 '이미지 실제 확인' = 시각근거 채점이 작동. 인스턴스는 이미지 없이 생성되어 약할 것으로 예상)")

asyncio.run(main())
