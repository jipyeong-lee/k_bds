"""
convert_to_swift.py — HF 데이터셋(parquet) → ms-swift GRPO 멀티모달 jsonl 변환 (컨테이너 내 실행)

ms-swift GRPO 포맷(검증가능/개방형 공통):
  {"messages":[{"role":"user","content":"<image>...질문"}], "images":["/abs/img.png"], "solution":"정답"}
  - 'solution' = ms-swift accuracy 보상(MathAccuracy)이 읽는 컬럼. 의료 개방형은 LLM judge 참조답으로 사용.
  - 이미지는 parquet 내장({bytes,path}) → 로컬 PNG 추출(컴퓨트노드 오프라인 대비), images 에 경로 기록.

지원 스키마 (자동 감지):
  - DeepVision-103K : question / images(list)   / reward_model.ground_truth  [RLVR]
  - medix-rl-data   : problem  / image(list)    / solution                   [개방형 의료]
  - MMK12           : question / image(dict)    / answer                     [RLVR·STEM, $$..$$ 정규화]
  - ThinkLite-VL    : problem(이미 <image> 포함) / image(raw bytes) / ground_truth [RLVR·hard]

* datasets 라이브러리 대신 pyarrow 로 스트리밍(저메모리 + datasets 버전 무관).

사용 (컨테이너 내):
  python scripts/convert_to_swift.py <name> --parquet <files...> --out <out.jsonl> --images-dir <dir> [--limit N]
"""
import argparse
import glob as globmod
import io
import json
import os

import pyarrow.parquet as pq
from PIL import Image


def to_pil(item):
    """parquet 이미지 항목(PIL / {'bytes','path'} / raw bytes)을 PIL.Image 로."""
    if isinstance(item, Image.Image):
        return item
    if isinstance(item, (bytes, bytearray)):           # ThinkLite-VL: 컬럼이 raw bytes
        return Image.open(io.BytesIO(item))
    if isinstance(item, dict):
        if item.get('bytes'):
            return Image.open(io.BytesIO(item['bytes']))
        if item.get('path') and os.path.exists(item['path']):
            return Image.open(item['path'])
    raise ValueError(f'unrecognized image item: {type(item)}')


def iter_rows(files):
    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', help='태그/스키마 힌트용 이름 (예: skylenage-ai/DeepVision-103K)')
    ap.add_argument('--parquet', nargs='+', required=True, help='로컬 parquet 파일/글롭')
    ap.add_argument('--out', required=True)
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--limit', type=int, default=0, help='0=전체')
    args = ap.parse_args()

    os.makedirs(args.images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    files = sorted(f for g in args.parquet for f in globmod.glob(g))
    if not files:
        raise SystemExit(f'parquet 파일 없음: {args.parquet}')
    # 스키마 감지(첫 파일 top-level 컬럼 — arrow 스키마라야 struct 가 평탄화 안 됨)
    cols = set(pq.ParquetFile(files[0]).schema_arrow.names)
    ans_col = None
    if 'reward_model' in cols:
        q_col, img_col, kind = 'question', 'images', 'deepvision'   # ans = reward_model.ground_truth
    elif {'problem', 'solution'} <= cols:
        q_col, ans_col, img_col, kind = 'problem', 'solution', 'image', 'medix'
    elif {'question', 'answer'} <= cols:                            # MMK12
        q_col, ans_col, img_col, kind = 'question', 'answer', 'image', 'mmk12'
    elif {'problem', 'answer'} <= cols:                             # ThinkLite-VL (problem 에 <image> 내장)
        q_col = 'problem'
        ans_col = 'ground_truth' if 'ground_truth' in cols else 'answer'
        img_col, kind = 'image', 'thinklite'
    else:
        raise SystemExit(f'알 수 없는 스키마. columns={sorted(cols)}')
    print(f'[{kind}] parquet {len(files)}개, q={q_col}, ans={ans_col or "reward_model"}, img={img_col} → {args.out}')

    tag = args.name.split('/')[-1].replace('.', '_')
    n = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        for i, row in enumerate(iter_rows(files)):
            if args.limit and n >= args.limit:
                break
            if kind == 'deepvision':
                sol = (row.get('reward_model') or {}).get('ground_truth')
            else:
                sol = row.get(ans_col)
            if sol is None or str(sol).strip() in ('', 'None'):
                continue
            sol = str(sol).strip()
            if kind in ('mmk12', 'thinklite'):
                sol = sol.strip('$').strip()               # $$10$$ → 10 (검증가능 평문화)
            imgs = row.get(img_col) or []
            if not isinstance(imgs, list):
                imgs = [imgs]
            paths = []
            for j, it in enumerate(imgs):
                try:
                    pim = to_pil(it).convert('RGB')
                except Exception:
                    continue
                p = os.path.abspath(os.path.join(args.images_dir, f'{tag}_{i}_{j}.png'))
                pim.save(p)
                paths.append(p)
            if not paths:
                continue
            q = str(row.get(q_col, '')).strip()
            # ThinkLite 의 problem 은 이미 <image> 토큰을 포함 → 중복 방지
            content = q if '<image>' in q else ('<image>' * len(paths) + '\n' + q)
            rec = {'messages': [{'role': 'user', 'content': content}],
                   'images': paths,
                   'solution': sol}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n += 1
            if n % 1000 == 0:
                print(f'  {n} written...')
    print(f'✅ {n} samples → {args.out}  (images: {args.images_dir})')


if __name__ == '__main__':
    main()
