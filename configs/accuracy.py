"""
accuracy.py — stage-2(DeepVision) 용 커스텀 정확도 보상 'accuracy_mix'

문제: 내장 MathAccuracy(math_verify)는 gold 가 수식으로 파싱 안 되면(예: 객관식 letter "B",
  기호 "∠2") gold_parsed 가 비어 reward=0 으로 스킵 → DeepVision 정답의 ~48%(객관식 letter)가
  정답이어도 항상 0점. 보상 신호 절반이 죽음.

해결: 정답 유형에 따라 분기.
  - 수식/숫자  → math_verify (내장과 동일 경로)
  - 객관식 letter(A~H) → letter 정규화 후 일치 비교
  - 기타 짧은 문자열 → 정규화 문자열 일치

또한 'format_think': 내장 Format(r'^<think>.*?</think>\s*<answer>.*?</answer>')은 .*? 라
  빈 <think></think>(비추론 지름길)도 1.0 을 줌 → 하이브리드 thinking 모델이 추론을 건너뛰고
  형식 보상만 챙기는 reward-hacking 발생. format_think 는 think 내부에 실질 추론(>=
  _MIN_THINK_CHARS)이 있어야만 1.0 → RL 이 항상-추론 모드로 수렴하도록 강제.

ms-swift 4.x 등록: from swift.rewards import ORM, orms; orms['accuracy_mix']=Cls
  사용: --external_plugins configs/accuracy.py --reward_funcs accuracy_mix format_think
       (+ --enable_thinking true 로 롤아웃 추론 템플릿 보장)
"""
import re
from typing import List

from swift.rewards import ORM, orms

_ANS_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
_BOXED_RE = re.compile(r'\\boxed\{(.*?)\}', re.DOTALL)
_LETTER_RE = re.compile(r'^[A-Ha-h]$')
# 예측에서 선택지 letter 추출: 맨 앞의 (B) / B. / B) / B: / 단독 B
_LETTER_PICK = re.compile(r'^\(?([A-Ha-h])\)?[\s.):]', re.DOTALL)


def _strip_answer(text):
    m = _ANS_RE.search(text)
    return (m.group(1).strip() if m else text.strip())


# === 태그 없는 출력의 letter 폴백 ======================================
# 왜 필요한가(사후분석 §4-4 · run 73924): `<answer>` 태그를 잃으면 위 `_strip_answer` 가
# **추론 전문(全文)** 을 답안으로 넘긴다. 그러면 `_LETTER_PICK` 이 `^` 앵커라 구조적으로
# 항상 0 점이 된다 → 형식을 잃는 순간 가중치 1.0 의 정확도까지 통째로 0 → advantage 가
# "정답이냐"가 아니라 "형식 지켰냐"로 지배됨 → 이탈이 이탈을 부르는 절벽.
# step 899 붕괴에서 의료(letter) −47.5pp 대 수학(본문 숫자 추출 가능) −6.5pp 로 갈린 게 이것이다.
#
# 실측으로 설계했다(`scripts/probe_answer_fallback.py`, 73924 롤아웃 9,504건):
#   · 붕괴 출력의 실제 모양은 "...corresponds to option B.\n</think><|im_end|>" —
#     **`</think>` 뒤가 비어 있다.** 그래서 뒤가 아니라 추론 **꼬리 자체**를 훑는다.
#   · 신호 어휘는 빈도 상위에서 뽑았다: "the correct answer is X" · "corresponds to option X"
#     · "the correct choice/option is X" · "matches option X".
#   · 태그를 인위로 제거한 검증: 복원 일치 **94.6%** · 거짓양성 **1.13%** · 놓침 4.3%.
#   · 순열검정 z=3.6~6.6 → 우연이 아니라 모델의 실제 답을 집는다.
#
# ⚠️ **의도적으로 불완전하다.** 회수는 개시 구간 정확도 0.000→0.236(정상 0.469의 절반).
#    형식을 잃으면 여전히 총보상의 ~48%를 잃는다(낙폭 −0.402 → −0.322). 목적은 절벽을
#    없애는 게 아니라 **경사로 바꾸는 것**이다 — 0 에 가까우면 형식을 버려도 손해가 없어진다.
_TAIL = 400
_SPECIAL = re.compile(r'<\|im_end\|>|<\|endoftext\|>|</?think>|</?answer>')
_CUE_FWD = re.compile(
    r'(?:answer|option|choice|정답|답)\s*(?:is|:|=|→|->|은|는)?\s*'
    r'[\*\'"“”\s(\[]*([A-H])(?![A-Za-z])', re.IGNORECASE)
# 역방향: "'B.(0, 0)' corresponds to this coordinate" — letter 가 신호 **앞**에 온다
_CUE_BWD = re.compile(
    r'(?<![A-Za-z])([A-H])[\)\.\'"”\s]*(?:is\s+the\s+)?'
    r'(?:corresponds|matches|is\s+correct|is\s+the\s+answer)', re.IGNORECASE)
# 최후 수단: 꼬리 끝에 홀로 선 **대문자** A~H. 소문자 a 는 영어 관사라 절대 받지 않는다.
_BARE_UPPER = re.compile(r'(?<![A-Za-z])\(?([A-H])\)?[\s.):,\'"]*$')


def _fallback_letter(text):
    """태그 없는 출력의 꼬리에서 최종 letter 를 뽑는다. None = 추출 실패(= 0점)."""
    t = _SPECIAL.sub(' ', text).strip()[-_TAIL:]
    for cand in reversed(_BOXED_RE.findall(t)):          # \boxed{B} 는 강한 신호
        c = cand.strip().strip('()[]{}').strip()
        if _LETTER_RE.match(c):
            return c.upper()
    hits = [(m.end(), m.group(1)) for m in _CUE_FWD.finditer(t)]
    hits += [(m.end(), m.group(1)) for m in _CUE_BWD.finditer(t)]
    if hits:
        return max(hits)[1].upper()                      # 가장 뒤의 신호를 쓴다
    m = _BARE_UPPER.search(t)
    return m.group(1).upper() if m else None


def _norm_pred(ans):
    """예측 최종답 정규화: <answer> 내용 → \\boxed 해제 → 괄호/공백/끝점 제거."""
    b = _BOXED_RE.search(ans)
    if b:
        ans = b.group(1).strip()
    ans = ans.strip().strip('()[]{}').strip()
    ans = ans.rstrip('.').strip()
    return ans


def _letter_match(pred_ans, gold_letter):
    p = _norm_pred(pred_ans)
    if _LETTER_RE.match(p):
        return float(p.upper() == gold_letter.upper())
    m = _LETTER_PICK.match(p)
    if m:
        return float(m.group(1).upper() == gold_letter.upper())
    return 0.0


class AccuracyMix(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        from math_verify import parse, verify  # noqa: F401  (존재 확인)

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from math_verify import parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            tagged = _ANS_RE.search(content) is not None   # 형식 유지 여부
            ans = _strip_answer(content)          # 모델 최종답 영역
            gold = _strip_answer(str(sol))        # 정답
            # 1) 수식/숫자 경로: gold 가 math 로 파싱되면 math_verify
            try:
                gold_parsed = parse(gold, extraction_mode='first_match')
            except Exception:
                gold_parsed = []
            if len(gold_parsed) != 0:
                try:
                    ans_parsed = parse(ans, extraction_mode='first_match')
                    rewards.append(float(verify(gold_parsed, ans_parsed)))
                except Exception:
                    rewards.append(0.0)
                continue
            # 2) 객관식 letter 경로
            if _LETTER_RE.match(gold):
                if tagged:
                    rewards.append(_letter_match(ans, gold))
                else:
                    # 태그 없음 → 전문을 넘기면 `^` 앵커 때문에 무조건 0 이다.
                    # 꼬리에서 최종 letter 를 찾는다. 못 찾으면 그대로 0.
                    got = _fallback_letter(content)
                    rewards.append(0.0 if got is None
                                   else float(got == gold.upper()))
                continue
            # 3) 기타 짧은 문자열: 정규화 일치
            rewards.append(float(_norm_pred(ans).casefold() == gold.casefold()))
        return rewards


orms['accuracy_mix'] = AccuracyMix


# === 추론-강제 format 보상 ============================================
# 롤아웃 시 템플릿이 thinking_prefix '<think>\n' 를 강제로 앞에 붙임. 보상 스코어링은
# 이 프리픽스를 포함한 텍스트로 평가됨(빈 think 가 내장 Format 에서 1.0 나오는 이유).
# 따라서 여기서도 동일 전제로, 프리픽스가 없으면 방어적으로 보강해 구조를 평가.
_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
_FULL_RE = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])',
                      re.DOTALL | re.MULTILINE)
_MIN_THINK_CHARS = 16        # think 내부 실질 추론 최소 길이(공백 제외). 빈/사소한 think 차단.


class FormatThink(ORM):
    """<think>(실질추론)</think><answer>...</answer> 구조 + 비어있지 않은 think 만 1.0."""

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for content in completions:
            text = content if content.lstrip().startswith('<think>') else '<think>\n' + content
            if not _FULL_RE.match(text):
                rewards.append(0.0)
                continue
            m = _THINK_RE.search(text)
            think = (m.group(1) if m else '')
            think = re.sub(r'\s+', '', think)        # 공백 제외 실질 길이
            rewards.append(1.0 if len(think) >= _MIN_THINK_CHARS else 0.0)
        return rewards


orms['format_think'] = FormatThink
