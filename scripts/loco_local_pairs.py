#!/usr/bin/env python3
"""loco_local_pairs.py — 풀이 줄이기 지점 단위 짝 시험(탐지기 없음). LoCo(arXiv 2611.04814)의 UniCut 변형이다.
기존 롤아웃에서 문항마다 가장 짧은 정답을 원본으로 잡고, 풀이를 빈 줄 단위로 나눠 첫 단계와 마지막 두 단계를 뺀 경계 중 균등 간격 k 곳에서 자른다.
자른 앞부분(prefix)을 assistant 글로 미리 채워 vLLM 에 n 번 이어 쓰게 하고, 정답이고 엄격 형식인 이어쓰기 중 가장 짧은 것을 좋은 답, 원래 연속을 나쁜 답으로 짝짓는다.
짝은 두 답을 prefix 부터 통째로 담는다. 두 답의 prefix 토큰은 DPO 의 로그비 차이에서 같은 값으로 빠지므로(정책·기준 모두 같은 문맥) prefix 조건 짝과 손실이 같다.
단, --rpo_alpha(좋은 답 NLL 항)를 켜면 그 항에는 prefix 가 들어간다.
사용: python3 -I scripts/loco_local_pairs.py <rollouts.jsonl> <이미지 dir> <출력 dir> [--url http://127.0.0.1:8000/v1] [--cuts 3] [--n 4] [--gap 200] [--n-q 20]
      --dry: 서버 없이 지점 수만 센다. --selftest: 자체 점검. 서버는 `vllm serve --served-model-name policy` 로 띄우되, Qwen3.5 채팅 템플릿은 마지막 assistant 글을 `</think>` 로 닫아 버리므로
      마지막 assistant 글을 그대로 두는 템플릿(--chat-template)이 필요하다. kgpu/loco_local.sh 가 패치해서 띄운다.
출력: <dir>/localpairs.jsonl(swift DPO 형식, 이미지 경로 IMG/<파일>) · images.txt"""
import argparse, asyncio, base64, json, os, re, statistics as S, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I 로 실행하면 스크립트 폴더가 경로에 없다
from fmt_strict import normalize, strict
SYS = ("You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, "
       "keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>.")
w = lambda t: len(t.split())


def form(t):  # loco_build_selfpairs.form 과 같다: 앞의 '<think>\n' 을 빼고 마지막 </answer> 뒤를 자른다. 형식이 맞지 않으면 None
    t = t[len('<think>\n'):] if t.startswith('<think>\n') else t
    i = t.rfind('</answer>')
    t = t[:i + len('</answer>')] if i >= 0 else t
    return t if t.count('</think>') == 1 and t.endswith('</answer>') and '<answer>' in t else None


def cut_points(think, k):
    """빈 줄 경계 중 첫 단계 뒤와 마지막 단계 앞을 뺀 것(= 첫 단계와 마지막 두 단계를 자르지 않는다)에서 균등 간격 k 곳. (prefix 글자 수 목록, 단계 수)."""
    b = [m.end() for m in re.finditer(r'\n\n+', think)]
    e = b[1:-1]
    return list(dict.fromkeys(e[j * len(e) // (k + 1)] for j in range(1, k + 1))) if e else [], len(b) + 1


def pick(rej, pre, valid, gap):
    """이어쓰기 중 가장 짧은 것이 원래 답(rej)보다 gap 단어 이상 짧을 때만 (좋은 답, 줄어든 단어 수). 아니면 None."""
    if not valid: return None
    g = min(valid, key=w)
    return (g, w(rej) - w(g)) if g.startswith(pre) and rej.startswith(pre) and w(rej) - w(g) >= gap else None


def sources(rows, k, n_q):
    """문항마다 가장 짧은 정답과 자를 지점 (r, src, rej, cuts, steps). src 는 엄격 형식으로 고친 것(이어 쓸 바탕), rej 는 고치기 전 원래 답(나쁜 답)이다. 단계가 4개 미만이면 건너뛴다."""
    out, skip = [], dict(no_source=0, few_steps=0)
    for r in rows:
        c = [(normalize(x), x) for x in (form(t) for t, ok, f in zip(r['completions'], r['correct'], r['finish']) if ok and f == 'stop') if x]
        c = [(n, x) for n, x in c if n]
        if not c: skip['no_source'] += 1; continue
        src, rej = min(c, key=lambda q: w(q[0])); cuts, steps = cut_points(rej.split('</think>')[0], k)   # 원래 풀이 안에서만 자른다(고치며 </think> 뒤 설명을 풀이로 옮긴 부분은 나쁜 답과 prefix 가 다르다)
        if not cuts: skip['few_steps'] += 1; continue
        out.append((r, src, rej, cuts, steps))
        if len(out) == n_q: break
    return out, skip


def selftest():
    cuts, steps = cut_points('a\n\nb\n\nc\n\nd\n\ne\n\nf', 3)
    assert steps == 6 and len(cuts) == 3 and cuts[0] == len('a\n\nb\n\n') and cuts[-1] == len('a\n\nb\n\nc\n\nd\n\n'), cuts   # 첫 경계(a 뒤)와 마지막 경계(e 뒤)는 빠진다
    assert cut_points('a\n\nb\n\nc', 3) == ([], 3) and cut_points('one step only', 3) == ([], 1)
    assert pick('p ' + 'x ' * 400 + '</answer>', 'p ', ['p ' + 'x ' * 100 + '</answer>'], 200)[1] == 300
    assert pick('p ' + 'x ' * 400, 'p ', ['p ' + 'x ' * 300], 200) is None and pick('p', 'p', [], 1) is None
    print('ok')


async def run(a, srcs):
    from openai import AsyncOpenAI
    from math_verify import parse, verify
    def ok(text, sol):   # kgpu/loco_rollouts.sh 와 같다: <answer> 안의 내용만 비교한다. 없으면 오답
        m = re.findall(r'<answer>(.*?)</answer>', text, re.S)
        if not m: return False
        x = m[-1].strip(); s = sol.strip()
        if x.lower() == s.lower(): return True
        try: return bool(verify(parse(f'${s}$'), parse(f'${x}$')))
        except Exception: return False
    cli = AsyncOpenAI(base_url=a.url, api_key='x', timeout=1800); sem = asyncio.Semaphore(a.sem)

    async def one(r, src, cut):
        img = 'data:image/png;base64,' + base64.b64encode(open(f"{a.images}/{r['image']}", 'rb').read()).decode()
        pre = src[:cut]
        async with sem:   # prefix 를 assistant 글로 미리 채워 이어 쓰게 한다
            c = await cli.chat.completions.create(model='policy', n=a.n, temperature=0.6, top_p=0.95, max_tokens=a.max_tok,
                messages=[{'role': 'system', 'content': SYS}, {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': img}}, {'type': 'text', 'text': r['question']}]},
                          {'role': 'assistant', 'content': '<think>\n' + pre.rstrip()}], extra_body={'continue_final_message': True, 'add_generation_prompt': False})   # 템플릿이 글을 trim 하므로 끝 빈 줄은 뺀다
        outs = [form(pre.rstrip() + (x.message.content or '')) for x in c.choices if x.finish_reason == 'stop']
        return len(c.choices), [g for g in outs if g and g.startswith(pre) and strict(g) and ok(g, r['solution'])], pre   # 이어쓰기가 빈 줄로 시작하지 않아 prefix 와 어긋나면 버린다
    jobs = [(r, src, rej, cut, steps) for r, src, rej, cuts, steps in srcs for cut in cuts]
    res = await asyncio.gather(*(one(r, src, cut) for r, src, _, cut, _ in jobs), return_exceptions=True)
    n = dict(resp=0, valid=0, small_gap=0, error=0); pairs = {}
    for (r, src, rej, cut, steps), x in zip(jobs, res):
        if isinstance(x, Exception): n['error'] += 1; print('오류', repr(x)[:160]); continue
        got, valid, pre = x; n['resp'] += got; n['valid'] += len(valid)
        p = pick(rej, pre, valid, a.gap)
        if not p: n['small_gap'] += 1; continue
        pairs.setdefault(r['id'], []).append({'id': r['id'], 'cut_chars': cut, 'steps': steps, 'saved': p[1],
            'messages': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': '<image>' + r['question']}, {'role': 'assistant', 'content': p[0]}],
            'rejected_response': rej, 'images': ['IMG/' + r['image']]})
    return [q for v in pairs.values() for q in sorted(v, key=lambda q: -q['saved'])[:a.max_pairs]], n


def main():
    if '--selftest' in sys.argv: return selftest()
    p = argparse.ArgumentParser(); p.add_argument('rollouts'); p.add_argument('images'); p.add_argument('out')
    p.add_argument('--url', default='http://127.0.0.1:8000/v1'); p.add_argument('--cuts', type=int, default=3); p.add_argument('--n', type=int, default=4)
    p.add_argument('--gap', type=int, default=200); p.add_argument('--n-q', type=int, default=20); p.add_argument('--max-pairs', type=int, default=3)
    p.add_argument('--max-tok', type=int, default=8192); p.add_argument('--sem', type=int, default=8); p.add_argument('--dry', action='store_true')
    a = p.parse_args()
    srcs, skip = sources(map(json.loads, open(a.rollouts)), a.cuts, a.n_q)
    print('원본 문항', len(srcs), '· 건너뜀', skip, '· 지점', sum(len(x[3]) for x in srcs), '· 원본 단어 중앙', S.median(w(x[1]) for x in srcs) if srcs else '-')
    if a.dry or not srcs: return
    pairs, n = asyncio.run(run(a, srcs)); os.makedirs(a.out, exist_ok=True)
    with open(f'{a.out}/localpairs.jsonl', 'w') as f:
        for q in pairs: f.write(json.dumps(q, ensure_ascii=False) + '\n')
    open(f'{a.out}/images.txt', 'w').write('\n'.join(sorted({q['images'][0][4:] for q in pairs})) + '\n')
    print(n, '→', len(pairs), '쌍 · 문항', len({q['id'] for q in pairs}), '· 줄어든 단어 중앙', S.median(q['saved'] for q in pairs) if pairs else '-')
    if pairs:   # 이어쓰기가 prefix 에 자연스럽게 붙었는지 눈으로 본다
        q = pairs[0]; i = os.path.commonprefix([q['messages'][2]['content'], q['rejected_response']]).rfind('\n\n') + 2
        print('--- 공유 prefix 끝\n', q['rejected_response'][max(0, i - 160):i], '\n--- 좋은 답 이어쓰기 앞\n', q['messages'][2]['content'][i:i + 160], '\n--- 나쁜 답 이어쓰기 앞\n', q['rejected_response'][i:i + 160])


if __name__ == '__main__':
    main()
