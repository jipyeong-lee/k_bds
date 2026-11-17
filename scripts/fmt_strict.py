"""fmt_strict.py — DPO 짝 생성 스크립트들이 같이 쓰는 엄격 형식 검사·정리. 규칙은 configs/accuracy.py 의 FormatThink 1.0 과 같다
(accuracy.py 는 swift 를, eval_v3_holdout.py 는 import 하면 평가가 돈다 — 그래서 따로 둔다). 답 t 는 생성 프롬프트가 붙인 '<think>\\n' 이 빠진 assistant 글.
왜: 풀이 줄이기 3~6차 짝의 좋은 답 100% 가 `</think>` 와 `<answer>` 사이에 설명을 둬서, 6차 모델이 그 버릇을 배웠다(2026-11-13).
"""
import re

FULL = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])', re.DOTALL)   # = accuracy.py _FULL_RE
MIN_THINK_CHARS = 16                                                                     # = accuracy.py _MIN_THINK_CHARS


def strict(t):
    """FormatThink 가 1.0 을 줄 답인가."""
    s = '<think>\n' + t
    m = re.search(r'<think>(.*?)</think>', s, re.DOTALL)
    return bool(m and len(re.sub(r'\s+', '', m.group(1))) >= MIN_THINK_CHARS and FULL.match(s))


def clean(t):
    """로그에 남은 <|im_end|> 와 앞의 '<think>' 를 뗀다."""
    t = t.replace('<|im_end|>', '').rstrip()
    return t[len('<think>'):].lstrip('\n') if t.startswith('<think>') else t


def normalize(t):
    """`</think>` 와 `<answer>` 사이 설명을 <think> 안으로 옮긴다(지우지 않는다 — 추론의 연장인 경우가 있다). 답 글자는 그대로.
    이미 엄격 형식이면 그대로, 고칠 수 없으면(태그 개수·끝이 어긋남) None."""
    t = clean(t)
    if t.count('</think>') != 1 or t.count('<answer>') != 1 or t.count('</answer>') != 1 or not t.endswith('</answer>'):
        return None
    think, rest = t.split('</think>')
    between, ans = rest.split('<answer>')
    g = think.rstrip() + ('\n\n' + between.strip() if between.strip() else '') + '\n</think>\n<answer>' + ans.strip()
    return g if strict(g) else None


if __name__ == '__main__':   # 자체 점검
    assert strict('reasoning that is long enough\n</think>\n<answer>A</answer>')
    assert not strict('reasoning that is long enough\n</think>\nThe answer is A.\n<answer>A</answer>')
    assert not strict('short\n</think>\n<answer>A</answer>')
    g = normalize('<think>\nreasoning that is long enough\n</think>\n\nThe answer is A.\n<answer>A</answer><|im_end|>')
    assert g == 'reasoning that is long enough\n\nThe answer is A.\n</think>\n<answer>A</answer>' and strict(g), g
    assert normalize('x\n</think>\n<answer>A</answer>\nextra') is None
    assert normalize('no close tag <answer>A</answer>') is None
    print('ok')
