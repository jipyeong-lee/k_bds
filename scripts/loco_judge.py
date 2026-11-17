#!/usr/bin/env python3
"""loco_judge.py — 5차 원본 답(work/loco5/d*.jsonl)의 끝까지 쓴 정답을 kgpu 의 큰 모델이 질로 채점한다. 표준 라이브러리만.
왜: 6차 짝은 '정답 중 가장 짧은 것'만 골라 풀이가 맞는지·군더더기가 없는지는 보지 않았다. 좋은 답을 질로 다시 고른다.
사용: KGPU_API_TOKEN=… python3 -I scripts/loco_judge.py <out.jsonl> [모델] [문항 수 20] [이미지 dir work/loco5/in/images] [동시 8]
  모델이 이미지를 못 읽으면(GLM·gpt-oss·DeepSeek-V4-Pro) 이미지를 빼고 글만 보낸다(TEXT_ONLY=1 로도 강제).
출력 행: id · model · idx(채점한 답 번호) · scores[] = {logic, perception, concise, overall, lucky} (1~5, lucky 는 정답이 근거 없이 맞았는지)
"""
import base64, concurrent.futures as cf, glob, json, os, random, re, sys, time, urllib.request

out = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else 'qwen3.8-flash-next-180b-a6b'
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
IMG = sys.argv[4] if len(sys.argv) > 4 else 'work/loco5/in/images'
PAR = int(sys.argv[5]) if len(sys.argv) > 5 else 8
TEXT_ONLY = os.environ.get('TEXT_ONLY') == '1' or MODEL.startswith(('glm', 'gpt-oss', 'deepseek-v4-pro'))   # 2026-11-17 kgpu 실측: 이 셋만 이미지를 거부
URL = 'https://api.kgpu.net/v1/chat/completions'; TOK = os.environ['KGPU_API_TOKEN']

RUBRIC = """You grade student solutions to a visual reasoning problem. All solutions below reached the correct final answer.
Grade EACH solution on a 1-5 scale:
- logic: every step is valid and the final answer follows from the steps (5 = fully sound; 1 = wrong reasoning that reaches the right answer by luck or a guess).
- perception: facts read from the image are correct ({perc}).
- concise: no repeated re-checking, no circular hesitation, no restating the task, no irrelevant detours (5 = tight; 1 = mostly filler or loops).
- overall: how good this solution is as a model answer to imitate.
- lucky: true if the right answer was reached despite wrong or missing reasoning.
Reply with ONLY a JSON object: {{"scores": [{{"logic": n, "perception": n, "concise": n, "overall": n, "lucky": bool}}, ...]}} in the same order as the solutions."""


def body(r, cands):
    text = f"Problem:\n{r['question']}\n\nCorrect answer: {r['solution']}\n\n" + '\n\n'.join(
        f'=== Solution {i + 1} ===\n{c}' for i, c in enumerate(cands))
    perc = 'you cannot see the image; judge consistency only and give 3 if unverifiable' if TEXT_ONLY else 'look at the image'
    user = text if TEXT_ONLY else [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(
        open(f"{IMG}/{r['image']}", 'rb').read()).decode()}}, {'type': 'text', 'text': text}]
    return json.dumps({'model': MODEL, 'temperature': 0, 'max_tokens': 16384, 'messages': [
        {'role': 'system', 'content': RUBRIC.format(perc=perc)}, {'role': 'user', 'content': user}]}).encode()


def judge(r):
    idx = [i for i, (f, k) in enumerate(zip(r['finish'], r['correct'])) if k and f == 'stop']
    b = body(r, [r['completions'][i] for i in idx]); err = ''
    for t in range(4):   # 502·JSON 깨짐이 가끔 온다
        try:
            req = urllib.request.Request(URL, b, {'Authorization': 'Bearer ' + TOK, 'Content-Type': 'application/json'})
            c = json.load(urllib.request.urlopen(req, timeout=900))
            s = json.loads(re.search(r'\{.*\}', c['choices'][0]['message']['content'] or '', re.S).group(0))['scores']
            assert len(s) == len(idx), f'점수 {len(s)}개 ≠ 답 {len(idx)}개'
            return {'id': r['id'], 'model': MODEL, 'idx': idx, 'scores': s, 'usage': c.get('usage')}
        except Exception as e:
            err = (e.read()[:300].decode('utf8', 'replace') if hasattr(e, 'read') else repr(e))[:300]; time.sleep(5 * (t + 1))
    return {'id': r['id'], 'model': MODEL, 'idx': idx, 'error': err}


rows = [json.loads(l) for f in sorted(glob.glob('work/loco5/d*.jsonl')) for l in open(f)]
have = set(os.listdir(IMG))
pool = [r for r in rows if r['image'] in have and sum(k and f == 'stop' for f, k in zip(r['finish'], r['correct'])) >= 2]
random.Random(0).shuffle(pool)
done = {json.loads(l)['id'] for l in open(out)} if os.path.exists(out) else set()
todo = [r for r in pool[:N] if r['id'] not in done]
print(f'후보 문항 {len(pool)} · 이번 {len(todo)} · 모델 {MODEL} · 글만 {TEXT_ONLY}', flush=True)
with open(out, 'a') as f, cf.ThreadPoolExecutor(PAR) as ex:
    for k, res in enumerate(ex.map(judge, todo), 1):
        f.write(json.dumps(res, ensure_ascii=False) + '\n'); f.flush()
        if k % 10 == 0 or 'error' in res: print(k, res.get('error', 'ok')[:200], flush=True)
