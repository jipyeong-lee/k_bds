#!/usr/bin/env python3
"""rubric_ablation.py — 루브릭 조합 C0~C4 를 학습 없이(고정 응답) 비교한다. 1단계 실험.

입력: <DIR>/rubric_rollouts.jsonl(kgpu/rubric_rollouts.sh 출력) · <DIR>/images/
채점: judge = kgpu deepseek-v4.1-flash(학습 judge 후보) · 기준 judge = kgpu medgemma-27b(다른 계열, 루브릭 없는 전체 평가 1~10)
대상: ① 롤아웃 100문항×8 ② 합성 변형 20문항×5(scripts/judge_audit.py 와 같은 문장)
지표: 변형 순서 · 기준 judge 와의 순위 상관(전체/그룹 안) · 학습 신호(그룹 안 점수가 다른 비율) · 정답(c1)과의 독립성 · 실패율·토큰
결과는 <DIR>/scores.jsonl · gold.jsonl 에 캐시(중단 후 다시 돌리면 이어서 한다) → <DIR>/summary.json
실행(저장소 루트): uv run -q --with openai --with math_verify==0.9.0 --with latex2sympy2_extended==1.11.0 \
                    python scripts/rubric_ablation.py work/rubric_ablation [--configs C0,C1,...] [--rows 100]
"""
import argparse, asyncio, json, math, os, statistics as st, sys, types

# swift 없이 보상 모듈을 쓰려고 스텁을 넣는다(scripts/test_medical_reward.py 와 같은 방식)
_sw, _swr = types.ModuleType('swift'), types.ModuleType('swift.rewards')
_swr.AsyncORM = _swr.ORM = type('ORM', (), {'__init__': lambda self, args=None, **kw: None}); _swr.orms = {}
_sw.rewards = _swr; sys.modules.setdefault('swift', _sw); sys.modules.setdefault('swift.rewards', _swr)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs'))
import medical_reward as M  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

JUDGE, GOLD = 'deepseek-v4.1-flash-ax-552b-a16b', 'medgemma-27b'
# judge_audit.py 의 합성 변형(같은 문장). good > vacuous_correct > halluc · good > wrong 이어야 한다
GROUNDED = ("On the provided image I localized the relevant region and examined its appearance "
            "(location, shape, margins, intensity) to derive the answer.")
VACUOUS = "I have read the question and will now state the final answer directly."
ABSURD = "The image clearly shows a sunny tropical beach with palm trees, blue sky and ocean waves."
WRONG = "an unrelated and incorrect answer"
VARIANTS = ('good', 'vacuous_correct', 'vacuous_wrong', 'wrong', 'halluc')
GOLD_PROMPT = ("You are an expert physician reviewing an AI answer to a medical image question. Using the IMAGE, the QUESTION and "
               "the REFERENCE ANSWER, rate the overall quality of the MODEL OUTPUT from 1 (worst) to 10 (best): correct final answer, "
               "reasoning faithful to what the image shows, no fabricated findings, no unasked claims.\n\nQUESTION: {q}\n"
               "REFERENCE ANSWER: {ref}\nMODEL OUTPUT: {out}\n\nRespond with ONLY a JSON object: {{\"score\": <1-10>}}.")


def variants(answer):
    mk = lambda t, x: f"<think>{t}</think><answer>{x}</answer>"
    return {'good': mk(GROUNDED, answer), 'vacuous_correct': mk(VACUOUS, answer), 'vacuous_wrong': mk(VACUOUS, WRONG),
            'wrong': mk(GROUNDED, WRONG), 'halluc': mk(ABSURD, answer)}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]: j += 1
        for k in range(i, j + 1): r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    """순위 상관(동률 평균 순위). 한쪽이 상수면 None."""
    if len(a) < 3: return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return None if den == 0 else sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den


def selftest():
    assert ranks([3, 1, 2]) == [2, 0, 1] and ranks([1, 1, 2]) == [0.5, 0.5, 2]
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9 and abs(spearman([1, 2, 3], [3, 2, 1]) + 1) < 1e-9
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    v = variants('X-ray'); assert all(M.format_ok(x) for x in v.values()) and 'X-ray' not in v['wrong']
    print('selftest OK')


def load_cache(path):
    return {json.loads(l)['_k']: json.loads(l) for l in open(path)} if os.path.exists(path) else {}


async def run(d, cfgs, n_rows):
    tok = next(l.split('=', 1)[1].strip().strip('"\'') for l in open('.env') if l.strip().removeprefix('export ').startswith('KGPU_API_TOKEN='))
    cli = AsyncOpenAI(base_url='https://api.kgpu.net/v1', api_key=tok, timeout=180, max_retries=3)
    rows = [json.loads(l) for l in open(f'{d}/rubric_rollouts.jsonl')][:n_rows]
    items = [(f"{r['id']}#{i}", r, c, 'rollout') for r in rows for i, c in enumerate(r['completions'])]
    items += [(f"{r['id']}@{n}", r, c, n) for r in rows[:20] for n, c in variants(r['answer']).items()]
    sc_path, gd_path = f'{d}/scores.jsonl', f'{d}/gold.jsonl'
    sc, gd = load_cache(sc_path), load_cache(gd_path)
    sem = asyncio.Semaphore(16)

    async def score(cfg, key, r, comp):
        k = f'{cfg}|{key}'
        if k in sc: return
        async with sem:
            res = await M.score_cfg(M.RUBRIC_CONFIGS[cfg], r['question'], r['solution'], comp, f"{d}/images/{r['image']}", cli, JUDGE)
        sc[k] = {'_k': k, 'cfg': cfg, 'key': key, **res}
        with open(sc_path, 'a') as f: f.write(json.dumps(sc[k], ensure_ascii=False) + '\n')

    async def gold(key, r, comp):
        if key in gd: return
        content = [{'type': 'text', 'text': GOLD_PROMPT.format(q=r['question'], ref=r['solution'], out=comp.strip())},
                   {'type': 'image_url', 'image_url': {'url': M._image_to_data_url(f"{d}/images/{r['image']}")}}]
        async with sem:
            try:
                resp = await cli.chat.completions.create(model=GOLD, messages=[{'role': 'user', 'content': content}], temperature=0, max_tokens=64)
                txt = M._msg_text(resp); m = M._JSON_RE.search(txt)
                s = float(json.loads(m.group(0))['score']) if m else None
            except Exception as e:
                s, txt = None, f'api:{type(e).__name__}'
        gd[key] = {'_k': key, 'score': s, 'raw': None if s is not None else txt[:200]}
        with open(gd_path, 'a') as f: f.write(json.dumps(gd[key], ensure_ascii=False) + '\n')

    jobs = [score(c, k, r, comp) for c in cfgs for k, r, comp, _ in items] + [gold(k, r, comp) for k, r, comp, _ in items]
    print(f'채점 대상 {len(items)} × 조합 {len(cfgs)} + 기준 judge {len(items)} = 요청 최대 {len(jobs)} (캐시 {len(sc) + len(gd)})', flush=True)
    for i in range(0, len(jobs), 400):   # 진행 표시
        await asyncio.gather(*jobs[i:i + 400]); print(f'  {min(i + 400, len(jobs))}/{len(jobs)}', flush=True)
    return summarize(items, cfgs, sc, gd)


def summarize(items, cfgs, sc, gd):
    out = {}
    for cfg in cfgs:
        get = lambda k: sc.get(f'{cfg}|{k}', {})
        roll = [(k, r['id']) for k, r, _, kind in items if kind == 'rollout']
        rw = {k: get(k).get('reward') for k, _ in roll}
        err = [get(k).get('err') for k, *_ in items if get(k).get('err') not in (None, 'format')]
        pairs = [(rw[k], gd[k]['score']) for k, _ in roll if k in gd and gd[k]['score'] is not None and rw[k] is not None]
        groups = {}
        for k, gid in roll: groups.setdefault(gid, []).append(k)
        within = [spearman([rw[k] for k in ks if gd.get(k, {}).get('score') is not None],
                           [gd[k]['score'] for k in ks if gd.get(k, {}).get('score') is not None]) for ks in groups.values()]
        within = [x for x in within if x is not None]
        signal = sum(1 for ks in groups.values() if len({rw[k] for k in ks}) > 1) / max(1, len(groups))
        c1 = [(get(k).get('v', {}).get('c1'), rw[k]) for k, _ in roll if get(k).get('v')]
        var = [x for x in (rw[k] for k, _ in roll) if x is not None]
        # c1 로 설명되지 않는 분산 비율 = 1 - R²(보상 ~ c1). 0 이면 루브릭이 정답 여부와 다를 바 없다
        r2 = None
        if c1 and len(var) > 1 and st.pvariance([x for _, x in c1]) > 0:
            groups_c1 = {}
            for c, x in c1: groups_c1.setdefault(c, []).append(x)
            within_ss = sum(sum((x - st.mean(xs)) ** 2 for x in xs) for xs in groups_c1.values())
            r2 = 1 - within_ss / (st.pvariance([x for _, x in c1]) * len(c1))
        var_means = {n: round(st.mean([get(k)['reward'] for k, _, _, kind in items if kind == n and get(k)]), 3)
                     for n in VARIANTS if any(kind == n and get(k) for k, _, _, kind in items)}
        sp = spearman(*zip(*pairs)) if len(pairs) > 2 else None
        out[cfg] = {
            'n_rollout': len(roll), 'mean_reward': round(st.mean(var), 3) if var else None,
            'variants': var_means,
            'order_ok': bool(var_means) and var_means.get('good', 0) > var_means.get('vacuous_correct', 1) > var_means.get('halluc', 2) - 1e-9
                        and var_means.get('good', 0) > var_means.get('wrong', 1),
            'spearman_gold_pooled': round(sp, 3) if sp is not None else None,
            'spearman_gold_within_group_mean': round(st.mean(within), 3) if within else None,
            'signal_groups_nonconstant': round(signal, 3),
            'beyond_c1_variance': round(1 - r2, 3) if r2 is not None else None,
            'fail_rate': round(len(err) / max(1, len(items)), 4), 'fail_kinds': sorted(set(err))[:5],
            'mean_tokens': round(st.mean([get(k).get('tokens') or 0 for k, *_ in items if get(k)]), 1),
        }
    g = [gd[k]['score'] for k, *_ in items if k in gd and gd[k]['score'] is not None]
    out['_gold'] = {'model': GOLD, 'n': len(g), 'mean': round(st.mean(g), 2) if g else None, 'fail': sum(1 for k, *_ in items if k in gd and gd[k]['score'] is None)}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('dir', nargs='?'); ap.add_argument('--configs', default='C0,C1,C2,C3,C4')
    ap.add_argument('--rows', type=int, default=100); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest: selftest(); sys.exit(0)
    cfgs = a.configs.split(','); assert all(c in M.RUBRIC_CONFIGS for c in cfgs), cfgs
    s = asyncio.run(run(a.dir, cfgs, a.rows))
    json.dump(s, open(f'{a.dir}/summary.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(s, ensure_ascii=False, indent=1))
