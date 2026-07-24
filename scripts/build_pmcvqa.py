#!/usr/bin/env python3
"""build_pmcvqa.py — PMC-VQA(의료 MC VQA) → swift RLVR jsonl.

PMC-VQA 는 CSV(Question/Choice A-D/Answer_label) + 이미지 zip 구조라 parquet 변환기와 별도.
정답 = Answer_label(letter A~D) → accuracy_mix 의 letter 경로로 검증. 질문에 보기 4개를 붙여 제시.
**서브샘플 이미지만 zip 에서 선택추출**(전체 149K 추출 회피). letter 균형 층화(seed=42).

사용:
  python scripts/build_pmcvqa.py --n 20000 \
    --csv <train.csv> <train_2.csv> --zips <images.zip> <images_2.zip> \
    --out work/data/pmcvqa_train.jsonl --images-dir work/data/images/pmcvqa
"""
import csv, json, os, re, random, zipfile, io, collections, argparse
from PIL import Image
random.seed(42)
csv.field_size_limit(10 ** 7)

_CHOICE_RE = re.compile(r'^\s*[A-D]\s*[:.\)]\s*')     # " A:Diffuse.. " / "A. .." / "A) .." 접두 제거


def clean_choice(c):
    return _CHOICE_RE.sub('', str(c).strip()).strip()


def label_of(r):
    # train.csv: Answer_label=letter / train_2.csv: Answer=letter (Answer_label 없음)
    return (str(r.get('Answer_label') or '').strip() or str(r.get('Answer') or '').strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--csv', nargs='+', required=True)
    ap.add_argument('--zips', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--images-dir', required=True)
    args = ap.parse_args()

    rows = []
    for p in args.csv:
        if os.path.exists(p):
            rows += list(csv.DictReader(open(p)))
    valid = [r for r in rows if label_of(r) in ('A', 'B', 'C', 'D')
             and str(r.get('Question', '')).strip() and str(r.get('Figure_path', '')).strip()]
    print(f'[pmcvqa] CSV {len(rows):,} → 유효 MC {len(valid):,}')

    # letter 균형 층화 샘플
    bylet = collections.defaultdict(list)
    for r in valid:
        bylet[label_of(r)].append(r)
    per = args.n // 4
    samp = []
    for L in 'ABCD':
        random.shuffle(bylet[L])
        samp += bylet[L][:per]
    random.shuffle(samp)
    print(f'[pmcvqa] 샘플 {len(samp):,} (letter별 {per:,})')

    # zip 이름 인덱스 (basename → (zip_idx, 내부경로)) — namelist 는 중앙디렉토리라 빠름
    zips = [zipfile.ZipFile(z) for z in args.zips]
    namemap = {}
    for zi, z in enumerate(zips):
        for nm in z.namelist():
            b = os.path.basename(nm)
            if b and b not in namemap:
                namemap[b] = (zi, nm)
    print(f'[pmcvqa] zip 이미지 인덱스 {len(namemap):,}')

    os.makedirs(args.images_dir, exist_ok=True)
    n = miss = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in samp:
            key = os.path.basename(r['Figure_path'].strip())
            if key not in namemap:
                miss += 1; continue
            zi, nm = namemap[key]
            try:
                img = Image.open(io.BytesIO(zips[zi].read(nm))).convert('RGB')
            except Exception:
                miss += 1; continue
            outp = os.path.abspath(os.path.join(args.images_dir, os.path.splitext(key)[0] + '.png'))
            img.save(outp)
            q = str(r['Question']).strip()
            opts = '\n'.join(f'{L}. {clean_choice(r["Choice " + L])}'
                             for L in 'ABCD' if str(r.get('Choice ' + L, '')).strip())
            content = '<image>\n' + q + '\n' + opts
            f.write(json.dumps({'messages': [{'role': 'user', 'content': content}],
                                'images': [outp], 'solution': label_of(r)},
                               ensure_ascii=False) + '\n')
            n += 1
            if n % 2000 == 0:
                print(f'  {n:,} written...')
    print(f'[pmcvqa] ✅ {n:,} → {args.out}  (이미지없음 {miss})')


if __name__ == '__main__':
    main()
