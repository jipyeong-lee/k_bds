#!/usr/bin/env python3
"""opd_server_probe.py — ms-swift GKD 를 교사 서버 모드(--teacher_model_server)로 돌리면서 교사와 오간 것을 기록하고 판정한다.

  run   : python opd_server_probe.py run <PROBE_DIR> -- <swift rlhf 인자...>
          gkd_trainer 를 패치해 학생 쪽 input_ids·labels·raw messages 와 교사 요청 messages·응답 prompt_token_ids·꼬리 logprobs 를
          PROBE_DIR/{student,teacher}_<n>.json 으로 남기고 swift rlhf 를 같은 프로세스에서 실행한다.
  check : python opd_server_probe.py check <PROBE_DIR> <TOKENIZER_DIR>
          C1 교사 messages 의 마지막이 학생 답(assistant)인가 · C2 ms-swift 꼬리 맞춤(마지막 n 토큰)이 학생 응답 토큰과 같은가
          · C3 그 위치에서 학생 토큰이 교사 top-20 안에 드는 비율.
  selftest : 꼬리 비교 함수 점검.
배경: gkd_trainer.py(4.1.3) 는 교사 API 에 raw_inputs(=데이터셋 messages)를 보내고(:601), 응답 logprobs 의 끝 n 개를
학생 응답 위치에 붙인다(:663-670). 학생 답이 messages 에 없거나 템플릿이 달라 꼬리가 어긋나도 오류 없이 학습된다.
"""
import json, os, sys

TAIL_KEEP = 4096  # 교사 응답 꼬리에서 보관할 위치 수(파일 크기 제한). MAX_COMP 보다 커야 C3 를 전부 잰다
MAX_SHIFT = 4


def tail_report(resp, teacher_ids):
    """ms-swift 꼬리 맞춤 흉내: 교사 마지막 n 토큰 ↔ 학생 응답 n 토큰. shift>0 = 교사 끝에 토큰이 shift 개 더 붙은 경우."""
    n, L = len(resp), len(teacher_ids)
    def score(s):  # 교사 끝 s 개를 떼고 비교
        w = teacher_ids[L - n - s: L - s] if L - n - s >= 0 else []
        return sum(a == b for a, b in zip(resp, w))
    return {'n': n, 'match': score(0), 'best_shift': max(range(MAX_SHIFT + 1), key=score), 'best_match': max(map(score, range(MAX_SHIFT + 1)))}


def selftest():
    assert tail_report([5, 6, 7], [1, 2, 5, 6, 7]) == {'n': 3, 'match': 3, 'best_shift': 0, 'best_match': 3}
    r = tail_report([5, 6, 7], [1, 5, 6, 7, 9])  # 교사 끝에 토큰 하나가 더 붙으면 전부 한 칸 밀린다
    assert r['match'] == 0 and r['best_shift'] == 1 and r['best_match'] == 3, r
    print('selftest OK')


def text_of(msgs):
    return [{'role': m.get('role'), 'content': m['content'] if isinstance(m.get('content'), str) else
             [c.get('text', f"<{c.get('type')}>") for c in (m.get('content') or []) if isinstance(c, dict)]} for m in msgs or []]


def run(probe_dir, swift_args):
    os.makedirs(probe_dir, exist_ok=True)
    import swift.rlhf_trainers.gkd_trainer as G
    cnt = {'s': 0, 't': 0}

    orig_build = G._build_teacher_session
    def build():
        sess = orig_build(); post = sess.post
        def logged_post(url, json=None, **kw):
            r = post(url, json=json, **kw)
            try:
                d = r.json(); ch = (d.get('choices') or [{}])[0]
                ids = d.get('prompt_token_ids') or ch.get('prompt_token_ids') or []
                lps = d.get('prompt_logprobs') or ch.get('prompt_logprobs') or []
                k = max(0, len(ids) - TAIL_KEEP)
                rec = {'url': url, 'messages': text_of((json or {}).get('messages')), 'prompt_token_ids': ids, 'tail_start': k,
                       'tail_logprobs': [{t: v['logprob'] for t, v in p.items()} if p else None for p in lps[k:]]}
            except Exception as e:
                rec = {'url': url, 'error': f'{type(e).__name__}: {e}'}
            cnt['t'] += 1
            with open(f"{probe_dir}/teacher_{cnt['t']:03d}.json", 'w') as f: f.write(__import__('json').dumps(rec))
            return r
        sess.post = logged_post
        return sess
    G._build_teacher_session = build

    orig_fetch = G.GKDTrainer._fetch_teacher_logprobs_from_api
    def fetch(self, encoded_inputs, raw_inputs=None):
        src = encoded_inputs.get('_opsd_teacher_inputs') or encoded_inputs
        cnt['s'] += 1
        with open(f"{probe_dir}/student_{cnt['s']:03d}.json", 'w') as f:
            json.dump({'input_ids': src['input_ids'].tolist(), 'labels': src['labels'].tolist(),
                       'raw_messages': [text_of(r.get('messages')) for r in (raw_inputs or [])]}, f)
        return orig_fetch(self, encoded_inputs, raw_inputs=raw_inputs)
    G.GKDTrainer._fetch_teacher_logprobs_from_api = fetch

    sys.argv = ['swift-rlhf'] + swift_args
    from swift.pipelines import rlhf_main
    rlhf_main()


def check(probe_dir, tok_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    S = sorted(f for f in os.listdir(probe_dir) if f.startswith('student_'))
    T = sorted(f for f in os.listdir(probe_dir) if f.startswith('teacher_'))
    print(f'학생 기록 {len(S)} · 교사 요청 {len(T)} (per_device_batch 1 이면 같아야 한다)')
    ok = {'C1': 0, 'C2': 0}; inc = [0, 0]
    for sf, tf in zip(S, T):
        s, t = json.load(open(f'{probe_dir}/{sf}')), json.load(open(f'{probe_dir}/{tf}'))
        if 'error' in t: print(tf, '교사 응답 파싱 실패:', t['error']); continue
        resp = [i for i, l in zip(s['input_ids'][0], s['labels'][0]) if l != -100]
        resp_text, ids_t, n = tok.decode(resp), t['prompt_token_ids'], len(resp)
        last = t['messages'][-1] if t['messages'] else {}
        c1 = last.get('role') == 'assistant' and isinstance(last.get('content'), str) and last['content'][:200].strip()[:100] in resp_text
        tr = tail_report(resp, ids_t); c2 = tr['match'] == n
        ok['C1'] += c1; ok['C2'] += c2
        for j, tid in enumerate(resp):  # ms-swift 가 쓰는 위치 그대로(꼬리 맞춤)
            p = len(ids_t) - n + j - t['tail_start']
            d = t['tail_logprobs'][p] if 0 <= p < len(t['tail_logprobs']) else None
            if d: inc[0] += str(tid) in sorted(d, key=lambda x: -d[x])[:20]; inc[1] += 1
        print(f'{sf}: roles={[m["role"] for m in t["messages"]]} C1={c1} C2={c2} 꼬리={tr}'
              f'\n   학생 응답 시작 {resp_text[:80]!r}\n   학생 응답 끝 {resp_text[-40:]!r}'
              f'\n   교사 시퀀스 끝 {tok.decode(ids_t[-min(len(ids_t), 25):])!r}'
              f'\n   교사가 받은 마지막 메시지 {str(last.get("content"))[:80]!r}')
    n = min(len(S), len(T))
    print(f'판정: C1 교사가 학생 답을 받음 {ok["C1"]}/{n} · C2 꼬리 정렬 일치 {ok["C2"]}/{n} · C3 top-20 포함 {inc[0]}/{inc[1]} = {inc[0] / max(1, inc[1]):.4f}')


def run46(probe_dir, swift_args):
    """ms-swift 4.6.0 용. 교사 요청은 샘플에서 만들어지고 위치는 앞에서부터 맞춘다(gkd_helpers.assemble_teacher_output).
    조립이 끝난 교사 top-k 텐서에서 바로 C3 을 잰다 — 학습이 실제로 쓰는 값이라 정렬이 맞는지가 그대로 드러난다."""
    os.makedirs(probe_dir, exist_ok=True)
    import torch
    from swift.rlhf_trainers.gkd_trainer import GKDTrainer
    cnt = [0]
    orig = GKDTrainer._fetch_and_assemble_teacher_logprobs

    def wrapped(self, chunks):
        reqs = [r for c in chunks for r in c.get('_teacher_requests', [])]
        info = [{'roles': [m.get('role') for m in getattr(r, 'messages', [])],
                 'last_content_type': type((getattr(r, 'messages', None) or [{}])[-1].get('content')).__name__} for r in reqs]
        orig(self, chunks)
        for c in chunks:
            ids, lab = c['teacher_model_inputs']['input_ids'], c['teacher_model_inputs'].get('labels')
            top = c['gkd_batch'].teacher_topk_indices
            if lab is None or top is None: continue
            nxt, m = ids[:, 1:], (lab[:, 1:] != -100)          # 위치 p 의 교사 분포는 토큰 p+1 을 예측한다
            hit = (top[:, :-1, :20] == nxt.unsqueeze(-1)).any(-1)
            cnt[0] += 1
            with open(f'{probe_dir}/call_{cnt[0]:03d}.json', 'w') as f:
                json.dump({'requests': info, 'n_pos': int(m.sum()), 'n_in_top20': int((hit & m).sum())}, f)

    GKDTrainer._fetch_and_assemble_teacher_logprobs = wrapped
    sys.argv = ['swift-rlhf'] + swift_args
    from swift.pipelines import rlhf_main
    rlhf_main()


def check46(probe_dir):
    C = [json.load(open(f'{probe_dir}/{f}')) for f in sorted(os.listdir(probe_dir)) if f.startswith('call_')]
    roles = [r['roles'] for c in C for r in c['requests']]
    got = sum(1 for r in roles if r and r[-1] == 'assistant')
    n, h = sum(c['n_pos'] for c in C), sum(c['n_in_top20'] for c in C)
    print(f'호출 {len(C)} · 요청 {len(roles)} · 예: roles={roles[:1]} last={[r["last_content_type"] for c in C for r in c["requests"]][:1]}')
    print(f'판정46: C1 교사 요청에 학생 답(assistant) {got}/{len(roles)} · C3 학생 응답 위치의 다음 토큰이 교사 top-20 {h}/{n} = {h / max(1, n):.4f}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if cmd == 'selftest': selftest()
    elif cmd == 'run': run(sys.argv[2], sys.argv[sys.argv.index('--') + 1:])
    elif cmd == 'check': check(sys.argv[2], sys.argv[3])
    elif cmd == 'run46': run46(sys.argv[2], sys.argv[sys.argv.index('--') + 1:])
    elif cmd == 'check46': check46(sys.argv[2])
    else: sys.exit(__doc__)
