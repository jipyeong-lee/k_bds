#!/usr/bin/env python3
"""DeepVision-103K 층화(stratified) 홀드아웃 분리 — 재현용.

DeepVision-103K 에는 카테고리 라벨이 없으므로(messages/images/solution 뿐),
'정답 유형'을 math/visual-logic 의 신뢰 가능한 기계적 프록시로 사용한다:
  - math        = 정답이 수치(numeric) 또는 수식(symbolic)   (45,284개)
  - visual-logic = 정답이 객관식 단일 문자 A-E (MC)            (51,896개)
  - other        = 그 외(모호) → 홀드아웃에서 제외, 학습에만 사용 (6,323개)
각 층에서 1%(매 100번째, offset 50)를 홀드아웃으로 떼어낸다 → math 453 + vl 519 = 972.
나머지(other 전량 포함)는 trainonly 로 학습에 사용. 홀드아웃은 학습에 절대 미사용
(→ Stage-2 는 init 부터 trainonly 로 fresh 학습해야 누수 0).

출력: work/data/deepvision_holdout.jsonl (+_stratum 필드), work/data/deepvision103k_trainonly.jsonl
"""
import json, re, os

SRC = 'work/data/deepvision103k_train.jsonl'
HOLD = 'work/data/deepvision_holdout.jsonl'
TRAIN = 'work/data/deepvision103k_trainonly.jsonl'
STRIDE = 100   # 1%
OFFSET = 50


def classify(sol):
    s = str(sol).strip()
    if re.fullmatch(r'[A-EＡ-Ｅ]', s):
        return 'vl'
    if re.fullmatch(r'-?\d+(\.\d+)?(\s*(cm|mm|m|°|degrees?)?)?', s):
        return 'math'                                   # numeric
    if re.search(r'\\(sqrt|frac|pi|times|circ)|[√π=]|\^|\d/\d', s):
        return 'math'                                   # symbolic
    if len(s) <= 3 and re.fullmatch(r'[A-Za-z]+', s):
        return 'vl'
    return 'other'


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    cnt = {'math': 0, 'vl': 0, 'other': 0}
    hold = {'math': 0, 'vl': 0}
    with open(HOLD, 'w') as fh, open(TRAIN, 'w') as ft:
        for l in open(SRC):
            d = json.loads(l)
            cat = classify(d['solution'])
            if cat in ('math', 'vl'):
                c = cnt[cat]
                cnt[cat] += 1
                if c % STRIDE == OFFSET:
                    d['_stratum'] = cat
                    fh.write(json.dumps(d, ensure_ascii=False) + '\n')
                    hold[cat] += 1
                    continue
            else:
                cnt['other'] += 1
            ft.write(l)
    print(f"분류: math={cnt['math']} vl={cnt['vl']} other={cnt['other']}")
    print(f"홀드아웃: math={hold['math']} vl={hold['vl']} 합계={sum(hold.values())}")


if __name__ == '__main__':
    main()
