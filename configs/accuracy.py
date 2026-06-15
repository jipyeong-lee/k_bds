"""
accuracy.py — stage-2(DeepVision) 용 커스텀 정확도 보상 'accuracy_mix'

문제: 내장 MathAccuracy(math_verify)는 gold 가 수식으로 파싱 안 되면(예: 객관식 letter "B",
  기호 "∠2") gold_parsed 가 비어 reward=0 으로 스킵 → DeepVision 정답의 ~48%(객관식 letter)가
  정답이어도 항상 0점. 보상 신호 절반이 죽음.

해결: 정답 유형에 따라 분기.
  - 수식/숫자  → math_verify (내장과 동일 경로)
  - 객관식 letter(A~H) → letter 정규화 후 일치 비교
  - 기타 짧은 문자열 → 정규화 문자열 일치

ms-swift 4.x 등록: from swift.rewards import ORM, orms; orms['accuracy_mix']=Cls
  사용: --external_plugins configs/accuracy.py --reward_funcs accuracy_mix format
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
                rewards.append(_letter_match(ans, gold))
                continue
            # 3) 기타 짧은 문자열: 정규화 일치
            rewards.append(float(_norm_pred(ans).casefold() == gold.casefold()))
        return rewards


orms['accuracy_mix'] = AccuracyMix
