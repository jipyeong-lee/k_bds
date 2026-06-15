"""
convert_to_swift.py — HF 데이터셋(parquet) → ms-swift GRPO 멀티모달 jsonl 변환 (컨테이너 내 실행)

ms-swift GRPO 포맷(검증가능/개방형 공통):
  {"messages":[{"role":"user","content":"<image>...질문"}], "images":["/abs/img.png"], "solution":"정답"}
  - 'solution' = ms-swift accuracy 보상(MathAccuracy)이 읽는 컬럼. 의료 개방형은 LLM judge 참조답으로 사용.
  - 이미지는 parquet 내장({bytes,path}) → 로컬 PNG 추출(컴퓨트노드 오프라인 대비), images 에 경로 기록.

지원 스키마 (자동 감지):
  - DeepVision-103K : question / images(list) / reward_model.ground_truth   [RLVR]
  - medix-rl-data   : problem  / image(list)  / solution                    [개방형 의료]

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
    """parquet 이미지 항목({'bytes','path'} 또는 PIL)을 PIL.Image 로."""
    if isinstance(item, Image.Image):
        return item
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
    if 'reward_model' in cols:
        q_col, img_col, kind = 'question', 'images', 'deepvision'
    elif {'problem', 'solution'} <= cols:
        q_col, img_col, kind = 'problem', 'image', 'medix'
    else:
        raise SystemExit(f'알 수 없는 스키마. columns={sorted(cols)}')
    print(f'[{kind}] parquet {len(files)}개, q={q_col}, img={img_col} → {args.out}')

    tag = args.name.split('/')[-1].replace('.', '_')
    n = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        for i, row in enumerate(iter_rows(files)):
            if args.limit and n >= args.limit:
                break
            sol = (row.get('reward_model') or {}).get('ground_truth') if kind == 'deepvision' else row.get('solution')
            if not sol:
                continue
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
            content = '<image>' * len(paths) + '\n' + str(row.get(q_col, '')).strip()
            rec = {'messages': [{'role': 'user', 'content': content}],
                   'images': paths,
                   'solution': str(sol).strip()}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n += 1
            if n % 1000 == 0:
                print(f'  {n} written...')
    print(f'✅ {n} samples → {args.out}  (images: {args.images_dir})')


if __name__ == '__main__':
    main()
