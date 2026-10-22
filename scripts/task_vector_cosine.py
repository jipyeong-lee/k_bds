#!/usr/bin/env python3
"""task_vector_cosine.py — 전문가 LoRA 의 ΔW(=task vector)끼리 코사인 유사도를 모듈별로 잰다.

merging 전 진단: 코사인 ≈ 0 이면 전문가들이 서로 다른 방향을 바꿔 간섭이 작다 → 단순 task arithmetic 으로
충분할 가능성이 크다. 값이 크게 음수인 모듈이 많으면 TIES/DARE 처럼 부호 충돌을 다루는 기법이 필요하다.

ΔW = B·A 를 만들지 않고 r×r 로 정확히 계산한다:  ⟨B₁A₁, B₂A₂⟩_F = Σ (B₁ᵀB₂) ∘ (A₁A₂ᵀ),  ‖BA‖²_F = Σ (BᵀB) ∘ (AAᵀ)
(α/r 스케일은 코사인에서 약분된다.)

  python scripts/task_vector_cosine.py name1=ckpt_dir1 name2=ckpt_dir2 ... [--json out.json]
"""
import itertools, json, os, re, statistics as st, sys
from collections import defaultdict

from safetensors import safe_open


def load(path):
    mods = defaultdict(dict)
    with safe_open(os.path.join(path, 'adapter_model.safetensors'), 'pt') as f:
        for k in f.keys():
            stem, part = k.removeprefix('base_model.model.').split('.lora_')
            mods[stem][part[0]] = f.get_tensor(k).double()
    return mods


def inner(a, b):
    return float(((a['B'].T @ b['B']) * (a['A'] @ b['A'].T)).sum())


def main():
    args = [a for a in sys.argv[1:] if '=' in a]
    out = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    names = [a.split('=', 1)[0] for a in args]
    ad = {n: load(a.split('=', 1)[1]) for n, a in zip(names, args)}
    keys = sorted(set.intersection(*(set(m) for m in ad.values())))
    assert keys, '공통 모듈이 없다'
    norm = {n: {k: inner(ad[n][k], ad[n][k]) ** 0.5 for k in keys} for n in names}

    res = {'pairs': {}, 'norm': {n: sum(v ** 2 for v in norm[n].values()) ** 0.5 for n in names}}
    for x, y in itertools.combinations(names, 2):
        cos = {k: inner(ad[x][k], ad[y][k]) / (norm[x][k] * norm[y][k] + 1e-30) for k in keys}
        by_type = defaultdict(list)
        by_layer = defaultdict(list)
        for k, v in cos.items():
            by_type[k.rsplit('.', 1)[-1]].append(v)
            m = re.search(r'layers\.(\d+)\.', k)
            by_layer[int(m.group(1)) if m else -1].append(v)
        # 전체 = 모듈별 내적을 모두 더한 뒤 정규화(전체 task vector 사이의 코사인)
        tot = sum(cos[k] * norm[x][k] * norm[y][k] for k in keys) / (res['norm'][x] * res['norm'][y])
        res['pairs'][f'{x}|{y}'] = {
            'global': tot, 'mean': st.mean(cos.values()), 'min': min(cos.values()), 'max': max(cos.values()),
            'by_type': {t: st.mean(v) for t, v in sorted(by_type.items())},
            'by_layer': {l: st.mean(v) for l, v in sorted(by_layer.items())}}
        print(f'{x:>8} vs {y:<8} 전체 {tot:+.4f}  모듈평균 {st.mean(cos.values()):+.4f}  '
              f'범위 [{min(cos.values()):+.3f}, {max(cos.values()):+.3f}]')
    print('‖ΔW‖ (α/r 미적용):', {n: round(v, 2) for n, v in res['norm'].items()})
    if out:
        json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
