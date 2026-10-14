#!/usr/bin/env python3
"""medical_reward.py 순수 로직 + 비동기 경로 유닛테스트 (judge API 없이, mock judge).

swift.rewards 를 스텁으로 주입해 컨테이너 없이 실행:
  python3 scripts/test_medical_reward.py
검증: 측정형 감지 · 루브릭 구성 · 형식 게이트 · JSON 파싱 · explicit 집계 단조성 · async mock 채점.
"""
import sys, types, asyncio, os

# --- swift.rewards 스텁 주입 (실제 패키지 없이 import 가능하게) ---
_sw = types.ModuleType('swift'); _swr = types.ModuleType('swift.rewards')
class _AsyncORM:
    def __init__(self, args=None, **kw): self.args = args
_swr.AsyncORM = _AsyncORM
_swr.orms = {}
_sw.rewards = _swr
sys.modules['swift'] = _sw; sys.modules['swift.rewards'] = _swr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
import medical_reward as M  # noqa: E402

PASS = 0; FAIL = 0
def check(name, cond):
    global PASS, FAIL
    print(('  ok  ' if cond else ' FAIL ') + name)
    PASS += cond; FAIL += (not cond)

print("== 1. 측정형 감지 ==")
check("'28 x 27 mm' 측정형", M.is_measurement('28 x 27 mm'))
check("'X-ray' 비측정형", not M.is_measurement('X-ray'))
check("'Endothelial surface' 비측정형", not M.is_measurement('Endothelial surface'))
check("'2.8 cm' 측정형", M.is_measurement('2.8 cm'))

print("== 2. 루브릭 구성 (측정형=4항목 / 비측정형=3항목) ==")
rb_m = M.build_rubric('28 x 27 mm', True)
rb_n = M.build_rubric('X-ray', False)
check("측정형 루브릭 4항목", len(rb_m) == 4)
check("측정형에 c3(정밀도) 포함", any(i['key'] == 'c3' for i in rb_m))
check("비측정형 루브릭 3항목", len(rb_n) == 3)
check("비측정형에 c3 없음", not any(i['key'] == 'c3' for i in rb_n))
check("참조답 템플릿 주입(c1)", "28 x 27 mm" in rb_m[0]['description'])
check("가중치 Essential=5", rb_m[0]['weight'] == 5)

print("== 3. 형식 게이트 ==")
good = "<think>CT상 좌하엽에 경계가 분명한 다낭성 종괴가 보이며 크기를 측정함</think><answer>\\boxed{28 x 27 mm}</answer>"
bare = "\\boxed{28 x 27 mm}"
empty_think = "<think></think><answer>\\boxed{X-ray}</answer>"
check("정상 형식 통과", M.format_ok(good))
check("맨몸 boxed 거부", not M.format_ok(bare))
check("빈 think 거부", not M.format_ok(empty_think))

print("== 4. JSON 파싱 ==")
check("정상 JSON 파싱", M.parse_verdicts('{"c1":1,"c2":1,"c3":1,"c4":1}', rb_m) == {'c1':1.,'c2':1.,'c3':1.,'c4':1.})
check("여분텍스트+JSON 파싱", M.parse_verdicts('here: {"c1":1,"c2":0,"c3":1,"c4":1} done', rb_m) is not None)
check("키 누락 → None", M.parse_verdicts('{"c1":1}', rb_m) is None)
check("쓰레기 → None", M.parse_verdicts('no json here', rb_m) is None)

print("== 5. explicit 집계 단조성 (측정형, 분모=5+3+3+4=15) ==")
def agg(c1,c2,c3,c4): return round(M.aggregate(rb_m, {'c1':c1,'c2':c2,'c3':c3,'c4':c4}), 4)
r_full = agg(1,1,1,1); r_nothink = agg(1,0,1,1); r_imprecise = agg(1,1,0,1); r_wrong = agg(0,0,0,0)
print(f"   full={r_full} no-think={r_nothink} imprecise={r_imprecise} wrong={r_wrong}")
check("완전정답 = 1.00", r_full == 1.0)
check("think영상언급無 = 0.80", r_nothink == round(12/15,4))
check("부정밀 = 0.80", r_imprecise == round(12/15,4))
check("오답 = 0.00", r_wrong == 0.0)
check("단조성 full>no-think>wrong", r_full > r_nothink > r_wrong)

print("== 6. async 경로 (mock judge) ==")
class _FakeResp:
    def __init__(self, txt):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=txt))]
class _FakeClient:
    def __init__(self, txt): self._txt = txt
    @property
    def chat(self): return self
    @property
    def completions(self): return self
    async def create(self, **kw): return _FakeResp(self._txt)

async def run_async():
    rw = M.ClinicalJudgeReward()
    rw._client = _FakeClient('{"c1":1,"c2":1,"c3":1,"c4":1}')  # judge 가 전부 1
    comps = [good, bare]  # 1) 정상 2) 형식위반
    sols = ['28 x 27 mm', '28 x 27 mm']
    imgs = [None, None]   # SEND_IMAGE 여도 None 이면 텍스트만
    return await rw.__call__(comps, solution=sols, images=imgs)

scores = asyncio.run(run_async())
print(f"   scores={scores}")
check("정상답 judge=all1 → 1.0", abs(scores[0] - 1.0) < 1e-6)
check("형식위반 → judge생략 0.0", scores[1] == 0.0)

print("== 7. 이미지 형식 처리 (swift 는 images 를 dict{bytes,path} 로 넘김 — 스모크 실측) ==")
import tempfile, base64 as _b64
_PNG = _b64.b64decode(  # 최소 유효 PNG(1x1)
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
_tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False); _tmp.write(_PNG); _tmp.close()
check("str 경로 → data_url", (M._image_to_data_url(_tmp.name) or '').startswith('data:image/png;base64,'))
check("dict{path} → data_url (swift 실측 포맷)",
      (M._image_to_data_url({'bytes': None, 'path': _tmp.name}) or '').startswith('data:image'))
check("dict{bytes} → data_url", (M._image_to_data_url({'bytes': _PNG, 'path': None}) or '').startswith('data:image'))
check("빈 dict → None", M._image_to_data_url({'bytes': None, 'path': None}) is None)
check("없는 경로 → None", M._image_to_data_url('/no/such/file.png') is None)
os.unlink(_tmp.name)

print("== 8. 증거 인식 크레딧 (Vacuous Credit 차단, RUBRIC_EVIDENCE) ==")
ALL1 = {'c1': 1., 'c2': 1., 'c3': 1., 'c4': 1.}
EV = "CT상 좌하엽에 경계가 분명한 다낭성 종괴가 보이며"
g = lambda ev, v=ALL1: M.apply_evidence_gate(v, ev, good)
check("인용이 think 에 실재 → c2·c4 유지", g(EV) == ALL1)
check("공백·대소문자 달라도 실재하면 유지", g("ct상  좌하엽에\n경계가 분명한 다낭성 종괴") == ALL1)
check("인용 없음 → c2=0, c4=0", g('')['c2'] == 0 and g('')['c4'] == 0 and g('')['c1'] == 1)
check("think 에 없는 지어낸 인용 → c2=0, c4=0", g("우하엽에 석회화 결절이 뚜렷함")['c4'] == 0)
check("너무 짧은 인용 → 거부", g("종괴")['c2'] == 0)
check("c2 가 이미 0 이면 c4 도 0", g(EV, {**ALL1, 'c2': 0.})['c4'] == 0)
check("c3 는 건드리지 않음", g('')['c3'] == 1)
check("프롬프트: 기본은 인용 지시 없음", 'c2_evidence' not in M.build_judge_prompt('q', 'r', 'a', rb_m))
check("프롬프트: evidence=True 면 인용 지시", 'c2_evidence' in M.build_judge_prompt('q', 'r', 'a', rb_m, evidence=True))
check("parse_evidence", M.parse_evidence('{"c1":1,"c2":1,"c2_evidence":"abc def"}') == 'abc def')
check("parse_evidence 필드 없음 → ''", M.parse_evidence('{"c1":1}') == '')

async def run_gate(judge_txt):
    rw = M.ClinicalJudgeReward(); rw._client = _FakeClient(judge_txt)
    return (await rw.__call__([good], solution=['28 x 27 mm'], images=[None]))[0]

M.EVIDENCE_GATE = True
try:
    s_ev = asyncio.run(run_gate('{"c1":1,"c2":1,"c3":1,"c4":1,"c2_evidence":"%s"}' % EV))
    s_no = asyncio.run(run_gate('{"c1":1,"c2":1,"c3":1,"c4":1}'))   # judge 가 인용 필드를 안 지킴
finally:
    M.EVIDENCE_GATE = False
print(f"   gate ON: 인용 있음={s_ev} / 인용 없음={s_no}")
check("게이트 ON + 유효 인용 → 1.0", abs(s_ev - 1.0) < 1e-6)
check("게이트 ON + 인용 누락 → 감점(c1·c3 만 = 8/15)", abs(s_no - 8/15) < 1e-6)
check("게이트 OFF(기본) 는 기존 동작 — judge all1 → 1.0", abs(asyncio.run(run_gate('{"c1":1,"c2":1,"c3":1,"c4":1}')) - 1.0) < 1e-6)

print(f"\n== 결과: {PASS} pass / {FAIL} fail ==")
sys.exit(1 if FAIL else 0)
