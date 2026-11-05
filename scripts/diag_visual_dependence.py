#!/usr/bin/env python3
"""diag_visual_dependence.py — 모델이 자기 답변 토큰을 만들 때 이미지를 얼마나 쓰는가(VPPO 진단).

VPPO(arXiv 2510.09285, VPPO-RL/verl/workers/actor/dp_actor.py:403-405)와 같은 추정량을 쓴다:
  d_t = log π(y_t | 가려진 이미지) − log π(y_t | 원본 이미지),  S_t = clamp(exp(d)−d−1, 0, 10)
가려진 이미지 = 14×14 패치를 확률 0.5 로 검게(perc_utils.random_patch_blackening).
답변은 모델이 원본 이미지로 직접 생성(temp 0)하고, 같은 답변을 두 이미지로 teacher-forced 채점한다(prompt_logprobs).
env: EVAL_BASE_URL EVAL_MODEL EVAL_DATA SYSTEM_PROMPT TOKENIZER(모델 폴더) DIAG_N(소스당, 100) DIAG_MAXTOK(4096)
     DIAG_CONC(16) DIAG_OUT(json) DIAG_SEED(0)
"""
import asyncio, base64, io, json, math, os, random, statistics as st, sys
import numpy as np
from PIL import Image

TOP_P = 0.4  # VPPO 7B 레시피(train_vppo_7b.sh). 논문 기본 코드값은 0.2
# Qwen3.5 템플릿은 assistant 메시지를 </think> 로 쪼개 trim·재조립해서 continue_final_message 가 400 이 된다(kgpu 점검 2026-11-04).
# 마지막 assistant 만 원문 그대로 내보내고 system·user 는 원본 템플릿과 같게 렌더링한다. 서버에 --chat-template-content-format openai 필요.
TEMPLATE = (
    "{%- for m in messages %}{{- '<|im_start|>' ~ m.role ~ '\\n' }}"
    "{%- if m.content is string %}{%- set c = m.content %}{%- else %}{%- set c %}"
    "{%- for p in m.content %}{%- if 'image' in p or 'image_url' in p or p.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{%- elif 'text' in p %}{{ p.text }}{%- endif %}{%- endfor %}{%- endset %}{%- endif %}"
    "{%- if m.role == 'assistant' and loop.last %}{{- c }}{%- else %}{{- c | trim ~ '<|im_end|>\\n' }}{%- endif %}"
    "{%- endfor %}"
)


def blacken(img, rng, patch=14, p=0.5):
    a = np.array(img.convert('RGB'))
    h, w = a.shape[:2]
    for y in range(0, h, patch):
        for x in range(0, w, patch):
            if rng.random() < p:
                a[y:y + patch, x:x + patch] = 0
    return Image.fromarray(a)


def sens(lp_orig, lp_aug):
    d = np.clip(np.asarray(lp_aug) - np.asarray(lp_orig), -20, 20)
    return np.clip(np.exp(d) - d - 1, 0, 10)


def summarize(s):  # 토큰별 S 배열 하나 → 요약
    k = max(1, math.ceil(len(s) * TOP_P))
    top = np.sort(s)[::-1][:k].sum()
    return {'mean': float(s.mean()), 'top_share': float(top / s.sum()) if s.sum() > 0 else 0.0,
            'frac_gt_0.1': float((s > 0.1).mean())}


def selftest():
    rng = random.Random(0)
    img = Image.new('RGB', (280, 280), (255, 255, 255))
    black = np.array(blacken(img, rng)).reshape(-1, 3).max(axis=1) == 0
    assert 0.3 < black.mean() < 0.7                           # 약 절반이 검다
    assert float(sens([-1.0], [-1.0])[0]) == 0.0              # 같으면 0
    assert float(sens([-1.0], [-3.0])[0]) > 0                 # 달라지면 양수
    s = summarize(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0]))
    assert s['top_share'] == 1.0 and s['frac_gt_0.1'] == 0.2  # 상위 40% 에 질량이 다 있다
    print('selftest OK')


def data_url(img):
    b = io.BytesIO(); img.save(b, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


async def main():
    from openai import AsyncOpenAI
    import requests
    from transformers import AutoTokenizer
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
    import accuracy as A
    tok = AutoTokenizer.from_pretrained(os.environ['TOKENIZER'])
    base, model = os.environ['EVAL_BASE_URL'], os.environ['EVAL_MODEL']
    system, n_per = os.environ['SYSTEM_PROMPT'], int(os.environ.get('DIAG_N', '100'))
    maxtok, seed = int(os.environ.get('DIAG_MAXTOK', '4096')), int(os.environ.get('DIAG_SEED', '0'))
    rows = [json.loads(l) for l in open(os.environ['EVAL_DATA'])]
    by = {}
    for r in rows: by.setdefault(r.get('_source', 'all'), []).append(r)
    rng = random.Random(seed)
    pick = []
    for k in sorted(by):
        g = by[k][:]; rng.shuffle(g); pick += g[:n_per]
    cli = AsyncOpenAI(base_url=base, api_key='EMPTY')
    sem = asyncio.Semaphore(int(os.environ.get('DIAG_CONC', '16')))
    am = A.AccuracyMix()

    def score_pass(img_url, q, text, n):  # 동기(requests) — 스레드로 돌린다
        msgs = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': img_url}}, {'type': 'text', 'text': q}]},
                {'role': 'assistant', 'content': '<think>\n' + text}]  # 생성 프롬프트가 '<think>\n' 까지 붙이므로 답변은 그 뒤에서 시작한다
        r = requests.post(f'{base}/chat/completions', timeout=600, json={
            'model': model, 'messages': msgs, 'max_tokens': 1, 'temperature': 0, 'prompt_logprobs': 1,
            'chat_template': TEMPLATE, 'add_generation_prompt': False, 'continue_final_message': True, 'return_token_ids': True})
        if r.status_code >= 400: raise RuntimeError(f'{r.status_code} {r.text[:600]}')
        d = r.json()
        ids = d.get('prompt_token_ids') or d['choices'][0]['prompt_token_ids']
        lps = d.get('prompt_logprobs') or d['choices'][0]['prompt_logprobs']
        ids, lps = ids[-n:], lps[-n:]
        if tok.decode(ids)[-40:].strip() != text[-40:].strip():
            return None  # 재토큰화가 다르면 정렬이 틀린다 — 버리고 센다
        return [lp[str(t)]['logprob'] for t, lp in zip(ids, lps)]

    errs = []

    async def one(r):  # 문항 하나의 실패가 전체를 죽이지 않게 한다. 오류는 세어서 아래에서 보고한다
        try:
            return await one_inner(r)
        except Exception as e:
            errs.append(f'{type(e).__name__}: {str(e)[:300]}')
            return None

    async def one_inner(r):
        async with sem:
            q = r['messages'][0]['content'].replace('<image>', '').strip()
            img = Image.open(r['images'][0])
            out = await cli.chat.completions.create(model=model, temperature=0.0, max_tokens=maxtok, messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': data_url(img.convert('RGB'))}}, {'type': 'text', 'text': q}]}])
            text = out.choices[0].message.content or ''
            n = len(tok(text, add_special_tokens=False)['input_ids'])
            if n < 8: return None
            a = await asyncio.to_thread(score_pass, data_url(img.convert('RGB')), q, text, n)
            b = await asyncio.to_thread(score_pass, data_url(blacken(img, random.Random(seed))), q, text, n)
            if a is None or b is None: return None
            s = sens(a, b)
            ok = float(am([text], [str(r['solution'])])[0])
            return {'source': r.get('_source', 'all'), 'correct': ok, 'n_tok': n, **summarize(s)}

    res = [x for x in await asyncio.gather(*[one(r) for r in pick]) if x]
    print(f'사용 {len(res)}/{len(pick)} (정렬 실패·짧은 답 제외) 예외 {len(errs)}건' + (f' 첫 오류: {errs[0]}' if errs else ''))
    summ = {}
    for key in sorted({x['source'] for x in res}) + ['ALL']:
        for lab, f in (('all', lambda x: True), ('correct', lambda x: x['correct'] > 0.5), ('wrong', lambda x: x['correct'] <= 0.5)):
            g = [x for x in res if (key == 'ALL' or x['source'] == key) and f(x)]
            if g:
                summ[f'{key}/{lab}'] = {'n': len(g), **{m: round(st.mean(x[m] for x in g), 4) for m in ('mean', 'top_share', 'frac_gt_0.1', 'n_tok')}}
    for k, v in summ.items(): print(k, v)
    json.dump({'summary': summ, 'items': res}, open(os.environ.get('DIAG_OUT', 'logs/diag_visual_dependence.json'), 'w'), ensure_ascii=False)


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else asyncio.run(main())
