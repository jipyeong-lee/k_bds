"""loco_api_answers.py — 풀이 줄이기 B안: 큰 Qwen 모델(kgpu 호스팅 API)이 같은 문항에 답한다. 표준 라이브러리만.
env: MAX_TOK(8192) · REDO=1 이면 out 에 이미 있는 답 중 finish=='length' 인 것만 다시 받아 같은 자리에 바꿔 쓴다.
사용(pod 안): KGPU_API_TOKEN 은 pod env. python3 loco_api_answers.py <in dir(rows.jsonl+images/)> <out.jsonl> [모델] [N_GEN]
출력 행: rows 필드 + answers[N_GEN] = {reasoning, content, finish, correct}. 생각 부분(reasoning_content)은 content 와 따로 온다.
"""
import base64, concurrent.futures as cf, json, os, re, sys, time, urllib.request
src, out = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else 'qwen3.8-flash-next-180b-a6b'; N = int(sys.argv[4]) if len(sys.argv) > 4 else 3
URL = os.environ.get('KGPU_API_BASE', 'https://api.kgpu.net/v1').rstrip('/') + '/chat/completions'; TOK = os.environ['KGPU_API_TOKEN']
SYS = ("You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step, keeping the reasoning concise. "
       "Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>.")
def ok(text, sol):
    m = re.findall(r'<answer>(.*?)</answer>', text, re.S)
    return bool(m) and m[-1].strip().lower() == sol.strip().lower()   # shortcut: 글자·숫자 문자열 일치만 본다. 식 답은 math_verify 로 다시 채점
def call(r):
    img = 'data:image/png;base64,' + base64.b64encode(open(f"{src}/images/{r['image']}", 'rb').read()).decode()
    body = json.dumps({'model': MODEL, 'temperature': 0.6, 'top_p': 0.95, 'max_tokens': int(os.environ.get('MAX_TOK', 8192)), 'messages': [
        {'role': 'system', 'content': SYS}, {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': img}}, {'type': 'text', 'text': r['question']}]}]}).encode()
    for t in range(4):   # 502 가 가끔 온다
        try:
            req = urllib.request.Request(URL, body, {'Authorization': 'Bearer ' + TOK, 'Content-Type': 'application/json'})
            c = json.load(urllib.request.urlopen(req, timeout=600))['choices'][0]
            m = c['message']; content = m.get('content') or ''
            return {'reasoning': m.get('reasoning_content') or '', 'content': content, 'finish': c.get('finish_reason'), 'correct': ok(content, r['solution'])}
        except Exception as e:
            err = repr(e)[:150]; time.sleep(5 * (t + 1))
    return {'reasoning': '', 'content': '', 'finish': 'error', 'correct': False, 'error': err}
rows = [json.loads(l) for l in open(f'{src}/rows.jsonl')]
old = {r['id']: r['answers'] for r in map(json.loads, open(out))} if os.environ.get('REDO') else {}
jobs = [(i, k) for i in range(len(rows)) for k in range(N) if not old or old[rows[i]['id']][k]['finish'] == 'length']
res = {i: list(old[rows[i]['id']]) if old else [] for i in range(len(rows))}
with cf.ThreadPoolExecutor(16) as ex:
    for (i, k), a in zip(jobs, ex.map(lambda j: call(rows[j[0]]), jobs)):
        if old: res[i][k] = a
        else: res[i].append(a)
with open(out, 'w') as f:
    for i, r in enumerate(rows): f.write(json.dumps({**r, 'answers': res[i]}, ensure_ascii=False) + '\n')
A = [a for v in res.values() for a in v]
print(f'모델 {MODEL} · 문항 {len(rows)} · 응답 {len(A)} · 정답 {sum(a["correct"] for a in A)} · 오류 {sum(a["finish"] == "error" for a in A)} · 길이 상한 {sum(a["finish"] == "length" for a in A)}')
