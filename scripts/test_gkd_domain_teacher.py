#!/usr/bin/env python3
"""kgpu/gkd_domain_teacher.patch 유닛테스트 — 원본 ms-swift 4.1.3 gkd_trainer.py 에 패치를 적용한 사본에서
추가된 코드(__init__ 블록·_set_teacher_domain·log 두 줄)를 그대로 뽑아 실행한다(GPU·실모델 없이, CPU).
  python3 -I scripts/test_gkd_domain_teacher.py [원본 gkd_trainer.py]   # 기본: /tmp/swift413/src/swift/rlhf_trainers/gkd_trainer.py
  (원본은 `pip download ms-swift==4.1.3 --no-deps` 로 받은 wheel 을 풀어 얻는다. torch·transformers·peft·accelerate 필요)
T1 구조: 패치 적용, 헬퍼 위치·호출 순서, @profiling_decorator 가 training_step 에 남아 있는가
T2 _set_teacher_domain: 도메인 판별·오류·no-op     T3 tiny Llama + LoRA 3개: 전환 수치 정확성·누수·requires_grad
"""
import ast, copy, itertools, os, subprocess, sys, tempfile, textwrap, types
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, '..', 'kgpu', 'gkd_domain_teacher.patch')
SRC = sys.argv[1] if len(sys.argv) > 1 else '/tmp/swift413/src/swift/rlhf_trainers/gkd_trainer.py'
PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    print(('  ok  ' if cond else ' FAIL ') + name)
    PASS += bool(cond); FAIL += (not cond)

def raises(exc, fn, *a):
    try:
        fn(*a)
    except exc:
        return True
    except Exception:
        return False
    return False

if not os.path.exists(SRC):
    sys.exit(f'원본 gkd_trainer.py 없음: {SRC}\n'
             '받는 법: pip download ms-swift==4.1.3 --no-deps -d X && unzip X/ms_swift-4.1.3-*.whl swift/rlhf_trainers/gkd_trainer.py -d X/src\n'
             '그 뒤:   python3 -I scripts/test_gkd_domain_teacher.py X/src/swift/rlhf_trainers/gkd_trainer.py')
torch.set_num_threads(1)
tmp = tempfile.TemporaryDirectory()
d = tmp.name

# ───────── T1: 패치 적용 + 구조 ─────────
os.makedirs(f'{d}/swift/rlhf_trainers')
with open(SRC, encoding='utf-8') as f:
    pristine = f.read()
open(f'{d}/swift/rlhf_trainers/gkd_trainer.py', 'w', encoding='utf-8').write(pristine)
r = subprocess.run(['patch', '-p1', '-i', os.path.abspath(PATCH)], cwd=d, capture_output=True, text=True)
check('T1 patch -p1 적용 성공', r.returncode == 0)
src = open(f'{d}/swift/rlhf_trainers/gkd_trainer.py', encoding='utf-8').read()
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GKDTrainer')
meth = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
deco = lambda fn: [ast.unparse(x) for x in fn.decorator_list]
check('T1 training_step 에 @profiling_decorator 가 남아 있음', 'profiling_decorator' in deco(meth['training_step']))
check('T1 _set_teacher_domain 에는 데코레이터가 없음', deco(meth['_set_teacher_domain']) == [])
body = [s for s in meth['training_step'].body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
call_src = lambda s: ast.unparse(s)
check('T1 training_step: args=… 다음 문장이 _set_teacher_domain(inputs), 그 뒤에 _build_opsd_teacher_data',
      [call_src(s) for s in body[:3]] == ['args = self.args', 'self._set_teacher_domain(inputs)',
                                          'teacher_data = self._build_opsd_teacher_data(inputs)'])
check('T1 prediction_step 첫 문장이 _set_teacher_domain(inputs)', call_src(meth['prediction_step'].body[0]) == 'self._set_teacher_domain(inputs)')

lines = src.split('\n')
def span(fn): return fn.lineno - 1, fn.end_lineno
i0 = next(i for i, l in enumerate(lines) if l.strip() == 'self._teacher_domains = []')
i1 = next(i for i, l in enumerate(lines) if l.strip() == '# Initialize teacher model')
a, b = span(meth['__init__'])
check('T1 __init__ 블록이 GKDTrainer.__init__ 안에 있고 "Initialize teacher model" 앞에 위치', a <= i0 < i1 < b)
init_block = textwrap.dedent('\n'.join(lines[i0:i1]))
j0 = next(i for i, l in enumerate(lines[span(meth['log'])[0]:], span(meth['log'])[0]) if l.strip() == 'if self._teacher_domains:')
log_block = textwrap.dedent('\n'.join(lines[j0:j0 + 2]))
set_domain = {}
exec(textwrap.dedent(ast.get_source_segment(src, meth['_set_teacher_domain'], padded=True)), set_domain)
set_domain = set_domain['_set_teacher_domain']

# ───────── T2: _set_teacher_domain 단위 ─────────
class Rec:
    def __init__(s): s.calls = []
    def set_adapter(s, name, inference_mode=False): s.calls.append((name, inference_mode))

def me_with(domains, teacher=None, acc=None):
    me = types.SimpleNamespace(_teacher_domains=domains, teacher_model=teacher if teacher is not None else Rec(),
                               accelerator=acc or types.SimpleNamespace(unwrap_model=lambda m: m))
    if domains:
        me._teacher_domain_steps = defaultdict(int)
    return me

D = ['deepvision', 'mmk12', 'pmcvqa']
img = lambda dom, f='x.png': {'images': [f'/root/w/data/images/{dom}/{f}']}
me = me_with(D)
for dom in D:
    set_domain(me, [img(dom)])
check('T2-a 3개 도메인 경로 → 각 이름으로 set_adapter(…, inference_mode=True)',
      me.teacher_model.calls == [(dom, True) for dom in D])
set_domain(me, [img('deepvision')]); set_domain(me, [img('deepvision', 'y.png')])
check('T2-a 도메인별 누적 횟수 {dv:3, mm:1, pmc:1}', dict(me._teacher_domain_steps) == {'deepvision': 3, 'mmk12': 1, 'pmcvqa': 1})
before = len(me.teacher_model.calls)
check('T2-b 알 수 없는 경로(/medix/) → ValueError, 어댑터 안 바꿈',
      raises(ValueError, set_domain, me, [img('medix')]) and len(me.teacher_model.calls) == before)
check('T2-c 한 경로에 도메인 2개(/deepvision/…/mmk12/) → ValueError',
      raises(ValueError, set_domain, me, [{'images': ['/data/deepvision/mmk12/x.png']}]))
check('T2-d micro-batch 에 도메인 2개 → ValueError', raises(ValueError, set_domain, me, [img('deepvision'), img('mmk12')]))
me2 = me_with(D)
set_domain(me2, [{'images': [{'path': '/root/w/data/images/mmk12/x.png', 'bytes': None}]}])
check('T2-e images 가 {"path":…} 형태여도 판별', me2.teacher_model.calls == [('mmk12', True)])
check('T2-e images 의 path 가 None(바이트만) → ValueError(조용히 넘어가지 않음)',
      raises(ValueError, set_domain, me2, [{'images': [{'path': None, 'bytes': b'x'}]}]))
check('T2-f images 키 없음 → ValueError', raises(ValueError, set_domain, me_with(D), [{}]))
check('T2-f images 빈 목록 → ValueError', raises(ValueError, set_domain, me_with(D), [{'images': []}]))
check('T2-i 접두사만 같은 디렉터리(/deepvision_extra/)는 매치 안 됨 → ValueError',
      raises(ValueError, set_domain, me_with(D), [{'images': ['/data/images/deepvision_extra/x.png']}]))
me0 = me_with([])
set_domain(me0, [{}])
check('T2-g _teacher_domains 가 비면 no-op(호출·속성 접근 없음)', me0.teacher_model.calls == [] and not hasattr(me0, '_teacher_domain_steps'))
check('T2-h 빈 inputs → ValueError(도메인 0개)', raises(ValueError, set_domain, me_with(D), []))

# ───────── T3: tiny Llama + LoRA 3개 ─────────
try:
    from accelerate import Accelerator
    from accelerate.utils import is_peft_model
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError as e:
    sys.exit(f'T3 임포트 실패({e}) — T3 는 실행되지 않았다. torch 가 있는 python 에서 venv 를 만들 것:\n'
             '  uv venv --system-site-packages --python <python> V && uv pip install --python V/bin/python --no-deps peft==0.19.1 accelerate==1.13.0\n'
             '  V/bin/python -I scripts/test_gkd_domain_teacher.py\n'
             '⚠ --no-deps 를 빼면 uv 가 시스템 torch 를 무시하고 CUDA torch·nvidia 라이브러리 수 GB 를 새로 받는다(15분 넘게 걸린다)')

cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
torch.manual_seed(0)
base = LlamaForCausalLM(cfg).eval()
TM = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
paths = {}
for i, name in enumerate(D):
    torch.manual_seed(100 + i)
    m = get_peft_model(copy.deepcopy(base), LoraConfig(task_type='CAUSAL_LM', r=16, lora_alpha=32, target_modules=TM, init_lora_weights=False))
    paths[name] = f'{d}/ad/{name}'
    m.save_pretrained(paths[name])
spec = ','.join(f'{n}={p}' for n, p in paths.items())

def build(spec_value, teacher):
    ns = {'os': types.SimpleNamespace(environ={'GKD_TEACHER_DOMAIN_ADAPTERS': spec_value} if spec_value is not None else {}),
          'defaultdict': defaultdict, 'is_peft_model': is_peft_model, 'teacher_model': teacher, 'self': types.SimpleNamespace()}
    exec(init_block, ns)
    return ns

plain = copy.deepcopy(base)
ns0 = build(None, plain)
check('T3-0 spec 없음 → 교사 그대로·_teacher_domains == []·카운터 없음',
      ns0['teacher_model'] is plain and ns0['self']._teacher_domains == [] and not hasattr(ns0['self'], '_teacher_domain_steps'))
check('T3-0 spec 형식 오류("deepvision" 에 = 없음) → ValueError', raises(ValueError, build, 'deepvision', copy.deepcopy(base)))
check('T3-0 이미 PeftModel 인 교사(--reward_adapters 경로) → AssertionError',
      raises(AssertionError, build, spec, PeftModel.from_pretrained(copy.deepcopy(base), paths['deepvision'])))
check('T3-0 spec 가 있는데 교사 모델이 None(--teacher_model 누락) → AssertionError', raises(AssertionError, build, spec, None))
try:                                  # 빈 문자열은 교사가 있어도 "set but empty" 로 멈춰야 한다(다른 assert 로 우연히 멈추는 것과 구분)
    build('', copy.deepcopy(base)); empty_msg = None
except Exception as e:                # 다른 예외로 멈추면 FAIL 줄로 보이게 한다(크래시 방지)
    empty_msg = f'{type(e).__name__}: {e}'
check('T3-0 spec 가 빈 문자열(GKD_TEACHER_DOMAIN_ADAPTERS=) → AssertionError "set but empty"(조용히 무시하지 않음)',
      empty_msg is not None and empty_msg.startswith('AssertionError') and 'set but empty' in empty_msg)
ns_sd = build(None, None)
check('T3-0 spec 없음 + 교사 None(자기증류)은 영향 없이 통과', ns_sd['teacher_model'] is None and ns_sd['self']._teacher_domains == [])

ns = build(spec, copy.deepcopy(base))
acc = Accelerator(cpu=True)
teacher = acc.prepare_model(ns['teacher_model'], evaluation_mode=True)
teacher.eval()
check('T3-1 교사가 PeftModel, 어댑터 이름 == 도메인 3개, 도메인 순서 유지',
      isinstance(teacher, PeftModel) and sorted(teacher.peft_config) == sorted(D) and ns['self']._teacher_domains == D)
me = types.SimpleNamespace(_teacher_domains=ns['self']._teacher_domains, _teacher_domain_steps=ns['self']._teacher_domain_steps,
                           teacher_model=teacher, accelerator=acc)
n_switch = [0]                       # T3-6 의 기대값: 아래에서 전환을 부른 횟수를 직접 센다
def switch(inputs):
    n_switch[0] += 1
    set_domain(me, inputs)
ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(7))
def logits(model):
    with torch.no_grad():
        return model(input_ids=ids).logits.float()
ref = {n: logits(PeftModel.from_pretrained(copy.deepcopy(base), paths[n]).eval()) for n in D}

got = {}
for n in D:
    switch([img(n)])
    got[n] = logits(teacher)
for n in D:
    diff = (got[n] - ref[n]).abs().max().item()
    check(f'T3-2 set_adapter({n}) 로짓 == {n} 하나만 올린 참조 모델 (max abs diff {diff:.1e})', diff == 0.0)
gaps = {(a, b): (ref[a] - ref[b]).abs().max().item() for a, b in itertools.combinations(D, 2)}
check(f'T3-3 도메인 어댑터끼리 로짓이 다름 (최소 격차 {min(gaps.values()):.3f} > 1e-3)', min(gaps.values()) > 1e-3)
with teacher.disable_adapter():
    nob = logits(teacher)
gb = {n: (ref[n] - nob).abs().max().item() for n in D}
check(f'T3-3 어댑터 없는 베이스와도 다름 (최소 격차 {min(gb.values()):.3f} > 1e-3)', min(gb.values()) > 1e-3)
for n in ['mmk12', 'deepvision', 'pmcvqa', 'mmk12', 'deepvision']:
    switch([img(n)])
    last = logits(teacher)
check('T3-4 전환 열(mm→dv→pmc→mm→dv) 뒤 deepvision 로짓이 처음과 동일(상태 누수 없음)', torch.equal(last, got['deepvision']))
for n in D:
    switch([img(n)])
    trainable = [k for k, p in teacher.named_parameters() if 'lora_' in k and p.requires_grad]
    check(f'T3-5 set_adapter({n}, inference_mode=True) 뒤 LoRA 파라미터 requires_grad 전부 False', not trainable)
teacher.set_adapter('mmk12')   # 대조: inference_mode 기본값
flipped = [k for k, p in teacher.named_parameters() if 'lora_' in k and p.requires_grad]
check(f'T3-5 (대조) inference_mode 기본값으로 바꾸면 활성 어댑터가 trainable 이 됨 → 패치가 True 를 주는 이유 ({len(flipped)}개 텐서)', bool(flipped))
switch([img('mmk12')])
check('T3-5 (대조 후) 패치 경로로 다시 전환하면 requires_grad 가 False 로 복구',
      not [k for k, p in teacher.named_parameters() if 'lora_' in k and p.requires_grad])
check(f'T3-6 누적 횟수 == 전환 호출 수 ({dict(me._teacher_domain_steps)}, 호출 {n_switch[0]}회)',
      n_switch[0] > 0 and sum(me._teacher_domain_steps.values()) == n_switch[0])
logs = {}
exec(log_block, {'self': me, 'logs': logs})
check('T3-7 log() 추가 두 줄: logs 에 teacher_steps/<도메인> 3개가 누적 횟수로 들어감',
      logs == {f'teacher_steps/{n}': me._teacher_domain_steps[n] for n in D})

tmp.cleanup()
print(f'\n{PASS} passed, {FAIL} failed')
sys.exit(1 if FAIL else 0)
