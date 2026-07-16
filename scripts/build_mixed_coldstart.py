#!/usr/bin/env python
"""
build_mixed_coldstart.py — 일반+의료 혼합 콜드스타트 SFT 데이터 빌드 (컨테이너 내 실행)

기존 build_rft_coldstart.py(자기증류 727건, format_think=0.473) 를 대체.

핵심 설계 (모두 실측 근거):
  1) 합격 게이트 = format_think == 1.0.  구 파이프라인은 closed()(= '</think>' 포함 + <answer> 존재)
     라는 느슨한 검사만 해서 데이터의 50.1% 가 RL 형식보상 0점 → RL 이 600스텝 태우고도 0.425 정체.
     여기서는 configs/accuracy.py 의 FormatThink 를 그대로 복제해 게이트로 쓴다.
  2) 간결성 상한(MAX_CHARS). 콜드스타트의 존재 이유가 "장황→잘림→형식0" 고리 차단.
     MMFineReason(추론 중앙값 9~11K자)·DeepVision 자기수확(4.4K자, 37.5%가 6K 초과) 을 이 기준으로 탈락시켰다.
  3) OpenMedReason 정답 위치 편향 보정. 정답이 거의 항상 A 에 배치되어 있음
     (4지선다 A=77%, 5지선다 A=86%) → 이미지 안 보고 A 만 찍어도 77~86%.
     보기 셔플은 추론문 29.6% 가 "option B" 식으로 문자를 참조해 불가 → **정답 문자별 균형 샘플링**으로 해소.
  4) 질문 단위 train/val 분할. 구 파이프라인은 질문 정렬 후 stride 라 같은 질문의 형제가 양쪽에 →
     val loss 가 낙관적(4에폭 내내 0.2136~0.2149 평탄했던 원인).
  5) 소스/난이도 층화 + 고정 시드. 구 파이프라인의 "가장 짧은 것 3개" 선별은 객관식에서
     추론 없는 찍기(<think></think><answer>C</answer>)를 우선 선택 → 빈 think 4.8% 유입.

사용:
  run_py python scripts/build_mixed_coldstart.py --out-train work/data/sft_mixed_train.jsonl \
      --out-val work/data/sft_mixed_val.jsonl --images-dir work/data/images/coldstart_mixed
"""
import argparse
import collections
import glob
import io
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq
from PIL import Image

PROJ = '/home01/k252a01/kbds_project'
HUB = f'{PROJ}/work/hf_cache/hub'

SYSTEM = ('You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step '
          'INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE '
          '<answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>.')

# ── configs/accuracy.py 의 FormatThink 복제 (swift import 없이 게이트로 사용) ──
_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
_FULL_RE = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])', re.DOTALL)
_MIN_THINK_CHARS = 16


def format_think(content):
    """configs/accuracy.py::FormatThink 와 동일 판정. 1.0 아니면 버린다."""
    text = content if content.lstrip().startswith('<think>') else '<think>\n' + content
    if not _FULL_RE.match(text.strip()):
        return 0.0
    m = _THINK_RE.search(text)
    think = re.sub(r'\s+', '', m.group(1) if m else '')
    return 1.0 if len(think) >= _MIN_THINK_CHARS else 0.0


def wrap(think, answer):
    return f'<think>\n{think.strip()}\n</think>\n<answer>{answer.strip()}</answer>'


_BOXED_RE = re.compile(r'\\boxed\{(.*)\}', re.DOTALL)
_DISPLAY_RE = re.compile(r'^\\\[\s*|\s*\\\]$')


def clean_answer(v):
    """<answer> 내용을 프로젝트 컨벤션(평문 정답만)으로 정규화.
    VLAA 실측: 66% 가 \\boxed{}, 76% 가 \\[..\\] display 래퍼, 100% 가 앞뒤 공백 →
    OMR('B')·VWI('50°') 스타일과 어긋남. \\boxed 컨벤션은 이 프로젝트에서 폐기됨
    (SYSTEM_PROMPT 는 '정답만' 요구, AccuracyMix 는 평문 검증)."""
    v = v.strip()
    v = _DISPLAY_RE.sub('', v).strip()
    m = _BOXED_RE.search(v)
    if m:
        v = m.group(1).strip()
    return v.strip()


def norm(text):
    """모델 출력에 <think> 프리픽스가 없는 경우 복원 (채팅 템플릿 response_prefix 와 동일)."""
    t = text.strip()
    return t if t.lstrip().startswith('<think>') else '<think>\n' + t


def save_img(obj, out_dir, name):
    """parquet 내장 이미지({'bytes','path'} 또는 PIL) → 로컬 PNG. 계산노드 오프라인 대비."""
    try:
        if isinstance(obj, dict):
            if obj.get('bytes'):
                im = Image.open(io.BytesIO(obj['bytes']))
            elif obj.get('path') and os.path.exists(obj['path']):
                im = Image.open(obj['path'])
            else:
                return None
        elif isinstance(obj, Image.Image):
            im = obj
        else:
            return None
        p = os.path.join(out_dir, name)
        im.convert('RGB').save(p)
        return p
    except Exception:
        return None


MAX_ANSWER_CHARS = 200   # <answer> 에 산문 설명이 통째로 들어간 건 배제(VLAA 일부)


def emit(q, assistant, img_path, max_answer=MAX_ANSWER_CHARS):
    """공통 출력 스키마. 게이트 미통과면 None.
    <answer> 는 clean_answer 로 정규화 후 재조립 → 세 소스 스타일 통일."""
    if not img_path:
        return None
    m = _THINK_RE.search(assistant if assistant.lstrip().startswith('<think>') else '<think>\n' + assistant)
    ma = re.search(r'<answer>(.*?)</answer>', assistant, re.DOTALL)
    if not m or not ma:
        return None
    ans = clean_answer(ma.group(1))
    if not ans or len(ans) > max_answer:      # 빈 정답 / 산문 정답 배제
        return None
    assistant = wrap(m.group(1), ans)
    if format_think(assistant) < 1.0:
        return None
    q = q.replace('<image>', '').strip()
    return {
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': '<image>\n' + q},
            {'role': 'assistant', 'content': norm(assistant)},
        ],
        'images': [img_path],
    }


# ─────────────────────────── 어댑터 ───────────────────────────

def src_openmedreason(n, max_chars, rng, img_dir):
    """의료. reasoning 이 이미 <think>..</think><answer>X</answer> (format_think=1.0 실측).
    정답 위치 편향(A=77~86%) 때문에 **정답 문자별 균형 샘플링** 필수."""
    fs = sorted(glob.glob(f'{HUB}/datasets--neginb--OpenMedReason/snapshots/*/data/train-*.parquet'))
    opt_re = re.compile(r'^([A-E])\.\s', re.M)
    buckets = collections.defaultdict(list)  # 정답문자 -> [(shard, row_idx)]
    for fi, f in enumerate(fs):
        t = pq.read_table(f, columns=['question', 'reasoning', 'answer'])
        for ri, r in enumerate(t.to_pylist()):
            rs = (r.get('reasoning') or '').strip()
            if len(rs) > max_chars or format_think(rs) < 1.0:
                continue
            if len(opt_re.findall(r.get('question') or '')) != 4:   # 4지선다만 사용(균형 가능 풀)
                continue
            buckets[(r.get('answer') or '').strip()].append((fi, ri))
    letters = [l for l in 'ABCD' if buckets.get(l)]
    per = n // len(letters)
    cap = min(len(buckets[l]) for l in letters)
    take = min(per, cap)
    print(f'  [omr] 풀: ' + ' '.join(f'{l}={len(buckets[l]):,}' for l in letters) +
          f'  → 문자당 {take:,}건 균형추출 (편향 제거)')
    want = collections.defaultdict(set)
    for l in letters:
        for fi, ri in rng.sample(buckets[l], take):
            want[fi].add(ri)
    out = []
    for fi in sorted(want):
        t = pq.read_table(fs[fi])
        rows = t.to_pylist()
        for ri in sorted(want[fi]):
            r = rows[ri]
            p = save_img(r.get('image'), img_dir, f'omr_{fi}_{ri}.png')
            e = emit(r.get('question') or '', (r.get('reasoning') or '').strip(), p)
            if e:
                e['_src'] = 'openmedreason'
                out.append(e)
    return out


def src_visualwebinstruct(n, max_chars, rng, img_dir):
    """일반. answer(단계별 추론) + short_answer(검증가능) → 래핑. difficulty 층화."""
    fs = sorted(glob.glob(f'{HUB}/datasets--TIGER-Lab--VisualWebInstruct-verified/snapshots/*/**/*.parquet',
                          recursive=True))
    by_diff = collections.defaultdict(list)
    for fi, f in enumerate(fs):
        t = pq.read_table(f, columns=['question', 'answer', 'short_answer', 'difficulty'])
        for ri, r in enumerate(t.to_pylist()):
            a = (r.get('answer') or '').strip()
            sa = (r.get('short_answer') or '').strip()
            if not a or not sa or len(a) > max_chars:
                continue
            by_diff[r.get('difficulty')].append((fi, ri))
    diffs = sorted([d for d in by_diff if by_diff[d]])
    per = max(1, n // len(diffs))
    want = collections.defaultdict(set)
    got = 0
    for d in diffs:                      # 난이도 균등 → 쉬운 문제 편향 차단
        take = min(per, len(by_diff[d]))
        for fi, ri in rng.sample(by_diff[d], take):
            want[fi].add(ri)
        got += take
    print(f'  [vwi] difficulty 층화: ' + ' '.join(f'{d}={min(per, len(by_diff[d])):,}' for d in diffs))
    out = []
    for fi in sorted(want):
        rows = pq.read_table(fs[fi]).to_pylist()
        for ri in sorted(want[fi]):
            r = rows[ri]
            imgs = r.get('images') or []
            if not imgs:
                continue
            p = save_img(imgs[0], img_dir, f'vwi_{fi}_{ri}.png')
            e = emit(r.get('question') or '', wrap(r['answer'], r['short_answer']), p)
            if e:
                e['_src'] = 'visualwebinstruct'
                out.append(e)
    return out


def src_vlaa(n, max_chars, rng, img_dir, subsets):
    """일반. answer 가 이미 <think>..</think><answer>..</answer> (format_think=1.0 실측).
    이미지는 tar 로 이미 추출된 것만 사용(vg/coco 는 레포에 tar 자체가 없어 제외)."""
    f = glob.glob(f'{HUB}/datasets--UCSC-VLAA--VLAA-Thinking/snapshots/*/VLAA-Thinking-SFT-126K.json')[0]
    roots = {'clevr_math': f'{PROJ}/work/data/images/vlaa_clevr',
             'synthesis': f'{PROJ}/work/data/images/vlaa_synthesis'}
    pool = collections.defaultdict(list)
    for line in open(f):                       # 확장자는 .json 이지만 실체는 JSONL
        r = json.loads(line)
        img = r.get('image') or ''
        sub = str(img).split('/')[0]
        if sub not in subsets:
            continue
        a = str(r.get('answer') or '')
        if len(a) > max_chars or format_think(a) < 1.0:
            continue
        pool[sub].append((img, r.get('question') or '', a))
    out = []
    per = max(1, n // max(1, len(subsets)))
    for sub in subsets:
        cand = pool.get(sub, [])
        picked = 0
        for img, q, a in rng.sample(cand, min(len(cand), per * 3)):   # 이미지 유실 대비 3배수 시도
            if picked >= per:
                break
            p = os.path.join(roots[sub], img)
            if not os.path.exists(p):
                continue
            e = emit(q, a, p)
            if e:
                e['_src'] = f'vlaa_{sub}'
                out.append(e)
                picked += 1
        print(f'  [vlaa/{sub}] 풀 {len(cand):,} → 채택 {picked:,}')
    return out


# ─────────────────────────── main ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-train', default=f'{PROJ}/work/data/sft_mixed_train.jsonl')
    ap.add_argument('--out-val', default=f'{PROJ}/work/data/sft_mixed_val.jsonl')
    ap.add_argument('--images-dir', default=f'{PROJ}/work/data/images/coldstart_mixed')
    ap.add_argument('--n-omr', type=int, default=5000, help='의료(OpenMedReason), 정답문자 균형')
    ap.add_argument('--n-vwi', type=int, default=3000, help='일반(VisualWebInstruct), difficulty 층화')
    ap.add_argument('--n-vlaa', type=int, default=2500, help='일반(VLAA), 이미 목표 형식')
    ap.add_argument('--vlaa-subsets', default='synthesis,clevr_math')
    ap.add_argument('--max-chars', type=int, default=6000, help='간결성 상한(잘림 방지)')
    ap.add_argument('--val-ratio', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    os.makedirs(a.images_dir, exist_ok=True)
    rows = []
    print('[build] OpenMedReason (의료)...')
    rows += src_openmedreason(a.n_omr, a.max_chars, rng, a.images_dir)
    print('[build] VisualWebInstruct (일반)...')
    rows += src_visualwebinstruct(a.n_vwi, a.max_chars, rng, a.images_dir)
    print('[build] VLAA-Thinking (일반)...')
    rows += src_vlaa(a.n_vlaa, a.max_chars, rng, a.images_dir, a.vlaa_subsets.split(','))

    # 질문 단위 분할 — 같은 질문의 형제가 train/val 에 갈라지지 않게(구 파이프라인 버그)
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r['messages'][1]['content']].append(r)
    qs = sorted(byq)
    rng.shuffle(qs)
    n_val_q = max(1, int(len(qs) * a.val_ratio))
    val_q = set(qs[:n_val_q])
    train = [r for q in qs if q not in val_q for r in byq[q]]
    val = [r for q in val_q for r in byq[q]]
    rng.shuffle(train)

    for path, data in ((a.out_train, train), (a.out_val, val)):
        with open(path, 'w') as fh:
            for r in data:
                fh.write(json.dumps({k: v for k, v in r.items() if not k.startswith('_')},
                                    ensure_ascii=False) + '\n')

    print(f'\n[build] train={len(train):,}  val={len(val):,}  (질문 단위 분할, 고유질문 {len(qs):,})')
    c = collections.Counter(r['_src'] for r in rows)
    for k, v in c.most_common():
        print(f'   {k:22s} {v:6,}')
    ft = [format_think(r['messages'][2]['content']) for r in rows]
    print(f'\n[build] ★ format_think = {sum(ft)/len(ft):.3f}  (게이트 통과분만 기록 — 구 콜드스타트는 0.473)')
    print(f'[build] → {a.out_train}')


if __name__ == '__main__':
    main()
