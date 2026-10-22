#!/usr/bin/env python3
"""merge_experts.py 유닛테스트 — 합성 base·어댑터 2개로 수식 검증(GPU·실모델 없이).
  python3 scripts/test_merge_experts.py
검증: linear = W₀ + Σ wᵢ·Bᵢ·Aᵢ·(α/r) · LoRA 없는 텐서 불변 · 비-safetensors 파일 복사 · ties 부호 다수결 · breadcrumbs 양끝 절단+합.
"""
import json, os, subprocess, sys, tempfile

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    print(('  ok  ' if cond else ' FAIL ') + name)
    PASS += cond; FAIL += (not cond)

torch.manual_seed(0)
d = tempfile.mkdtemp()
K, N = 'model.language_model.layers.0.mlp.up_proj.weight', 'model.visual.patch.weight'
base = {K: torch.randn(6, 4).bfloat16(), N: torch.randn(3, 3).bfloat16()}
os.makedirs(f'{d}/base')
save_file(base, f'{d}/base/model-00001-of-00001.safetensors', metadata={'format': 'pt'})
json.dump({'weight_map': {K: 'model-00001-of-00001.safetensors', N: 'model-00001-of-00001.safetensors'}},
          open(f'{d}/base/model.safetensors.index.json', 'w'))
open(f'{d}/base/config.json', 'w').write('{}')

AB = []
for i in range(2):
    A, B = torch.randn(2, 4), torch.randn(6, 2)
    AB.append((A, B))
    os.makedirs(f'{d}/a{i}')
    stem = 'base_model.model.' + K.removesuffix('.weight')
    save_file({stem + '.lora_A.weight': A, stem + '.lora_B.weight': B}, f'{d}/a{i}/adapter_model.safetensors')
    json.dump({'r': 2, 'lora_alpha': 4}, open(f'{d}/a{i}/adapter_config.json', 'w'))

def run(method, out, density='1.0', *extra):
    subprocess.run([sys.executable, f'{HERE}/merge_experts.py', '--base', f'{d}/base', '--adapters', f'{d}/a0,{d}/a1',
                    '--weights', '0.5,1.0', '--method', method, '--density', density, '--out', f'{d}/{out}', *extra],
                   check=True, capture_output=True)
    return load_file(f'{d}/{out}/model-00001-of-00001.safetensors')

print('== linear ==')
m = run('linear', 'lin')
taus = [w * 2.0 * (B @ A) for (A, B), w in zip(AB, [0.5, 1.0])]
want = (base[K].float() + taus[0] + taus[1]).bfloat16()
check('W = W0 + Σ wᵢ·Bᵢ·Aᵢ·(α/r)', torch.equal(m[K], want))
check('LoRA 없는 텐서(비전) 불변', torch.equal(m[N], base[N]))
check('dtype bf16 유지', m[K].dtype == torch.bfloat16)
check('config.json 복사', os.path.exists(f'{d}/lin/config.json'))
check('merge_config.json 기록', json.load(open(f'{d}/lin/merge_config.json'))['modules'] == 1)

print('== ties ==')
t = run('ties', 'ties')[K].float() - base[K].float()
st = torch.stack(taus)
sign = torch.sign(st.sum(0)); agree = torch.sign(st) == sign
want_t = (st * agree).sum(0) / agree.sum(0).clamp(min=1)       # density 1.0 = 자르지 않음 → 정확한 기대값
check('density 1.0: 부호 다수결 + 합의값 평균', torch.allclose(t, want_t.bfloat16().float(), atol=0.05))
t2 = run('ties', 'ties2', '0.25')[K].float() - base[K].float()
check('density 0.25: 어댑터당 6칸만 남아 합집합 ≤ 12칸', int((t2 != 0).sum()) <= 12)

print('== breadcrumbs ==')
b = run('breadcrumbs', 'bc', '0.5', '--gamma', '0.25')[K].float() - base[K].float()
kept = []
for t in taus:                                    # 24칸 중 크기순 위 6칸·아래 6칸을 지우고 가운데 12칸만
    r = t.abs().flatten().argsort().argsort().view_as(t)
    kept.append(torch.where((r >= 6) & (r < 18), t, torch.zeros_like(t)))
sk = torch.stack(kept)
sgn = torch.where(sk.sum(0) >= 0, 1.0, -1.0)
want_b = (sk * (torch.sign(sk) == sgn)).sum(0)   # normalize false → 평균이 아니라 합
check('gamma 0.25·density 0.5: 양끝 절단 후 부호 합의값의 합', torch.allclose(b, want_b.bfloat16().float(), atol=0.05))
check('gamma 가 크기 최대값 칸을 지움', all(k.abs().max() < t.abs().max() for k, t in zip(kept, taus)))

print(f'\n{PASS} passed, {FAIL} failed')
sys.exit(1 if FAIL else 0)
