"""
medical_reward.py — [3단계] 의료 멀티모달 VQA 루브릭 보상 (Rubric-as-a-Reward / RaR)

ms-swift GRPO 의 --external_plugins 로 로드되어 'clinical_judge' 보상을 등록한다.
실행: swift rlhf ... --external_plugins configs/medical_reward.py \
                     --reward_funcs format_think clinical_judge

설계(상세 docs/medical_reward_spec.md §4):
  - 판정을 단일 스칼라가 아니라 **가중 다기준 체크리스트(RaR)** 로 분해.
    judge 가 항목별 0/1 → explicit 집계 r = Σ wⱼcⱼ / Σ wⱼ ∈ [0,1].
  - medix 는 단답 VQA(정답≈1사실) → **LLM 루브릭 생성 불필요**, 참조답을 Essential 기준에 템플릿 주입.
  - 정적 4차원: 정답정확성(5) / 시각근거(3, <think>) / 정밀도·단위(3, 측정형 한정) / 환각Pitfall(4).
  - 멀티모달 judge 가 이미지를 직접 보고 시각근거·환각을 검증(judge 의 핵심 강점).

judge 는 OpenAI 호환 엔드포인트(내부 self-host vLLM 또는 도달가능 API). env 로 주입:
  JUDGE_BASE_URL (예: http://<judge-node>:8100/v1) / JUDGE_MODEL / JUDGE_API_KEY
LLM-as-a-judge 는 I/O 바운드 → AsyncORM(asyncio.gather 병렬). 학습 GPU 미점유.

견고성: 타임아웃/실패/파싱불가 → 0.0(학습 중단 방지). 형식 위반 → judge 생략하고 0(비용↓·형식강제).
"""
import os
import re
import json
import base64
import asyncio
from typing import List, Optional

from swift.rewards import AsyncORM, orms

# ---- 설정 (env override) ---------------------------------------------------
JUDGE_BASE_URL = os.environ.get('JUDGE_BASE_URL', 'http://127.0.0.1:8100/v1')
JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'gemma4-26b')
JUDGE_API_KEY = os.environ.get('JUDGE_API_KEY', 'EMPTY')
JUDGE_TIMEOUT = float(os.environ.get('JUDGE_TIMEOUT', '60'))
JUDGE_MAX_TOKENS = int(os.environ.get('JUDGE_MAX_TOKENS', '512'))
MEASURE_TOL_PCT = int(os.environ.get('JUDGE_MEASURE_TOL_PCT', '15'))
SEND_IMAGE = os.environ.get('JUDGE_SEND_IMAGE', '1') == '1'   # judge 가 멀티모달이면 1
# Qwen3 계열 judge 는 추론모델 → JSON 판정만 필요하므로 thinking 끔(빈 content/토큰낭비 방지).
JUDGE_THINK = os.environ.get('JUDGE_THINK', '0') == '1'
_EXTRA_BODY = {} if JUDGE_THINK else {'chat_template_kwargs': {'enable_thinking': False}}


def _msg_text(resp) -> str:
    """추론모델 대응: content 비면 reasoning_content 폴백."""
    m = resp.choices[0].message
    return (getattr(m, 'content', None) or getattr(m, 'reasoning_content', None) or '')

# 카테고리 가중치 (RaR 정수 스킴)
W_ESSENTIAL, W_IMPORTANT, W_PITFALL, W_OPTIONAL = 5, 3, 4, 1

# ---- 정규식 --------------------------------------------------------------
_ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
# 형식 게이트: <think>(실질추론)</think> <answer>...</answer>
_FORMAT_RE = re.compile(r'<think>(.*?)</think>\s*<answer>(.*?)</answer>', re.DOTALL)
_MIN_THINK_CHARS = 15
# 측정형 감지: 숫자 + 의료 단위
_MEASURE_RE = re.compile(
    r'\d+(?:\.\d+)?\s*(?:mm|cm|millimet|centimet|cc|ml|mL|μm|um|nm|degree|°|%|HU)\b',
    re.IGNORECASE)
# judge 출력에서 JSON 추출
_JSON_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)


def is_measurement(reference: str) -> bool:
    return bool(_MEASURE_RE.search(reference or ''))


def build_rubric(reference: str, measurement: bool) -> List[dict]:
    """단답 VQA 용 정적 4차원 루브릭. 기준1 description 에 참조답 주입(템플릿)."""
    ref = (reference or '').strip()
    rubric = [
        {'key': 'c1', 'title': 'Answer correctness', 'weight': W_ESSENTIAL,
         'description': (f"Essential Criteria: The <answer> matches the reference answer "
                         f"'{ref}' in meaning (synonyms, unit conversion, and paraphrase are allowed).")},
        {'key': 'c2', 'title': 'Visual grounding', 'weight': W_IMPORTANT,
         'description': ("Important Criteria: The <think> reasoning cites ACTUAL visual findings in the "
                         "image (e.g. location, shape, boundary, intensity) that genuinely support the answer, "
                         "rather than generic or fabricated description.")},
    ]
    if measurement:
        rubric.append(
            {'key': 'c3', 'title': 'Numeric precision & unit', 'weight': W_IMPORTANT,
             'description': (f"Important Criteria: The numeric magnitude AND unit are within "
                             f"+/-{MEASURE_TOL_PCT}% of the reference '{ref}'.")})
    rubric.append(
        {'key': 'c4', 'title': 'No hallucination / overclaim', 'weight': W_PITFALL,
         'description': ("Pitfall Criteria (positively phrased): The answer does NOT introduce findings "
                         "absent from the image, and does NOT add claims beyond what the question asked.")})
    return rubric


def _strip_tag(text: str, regex) -> str:
    m = regex.search(text or '')
    return (m.group(1).strip() if m else (text or '').strip())


def format_ok(completion: str) -> bool:
    """think 내부에 실질 추론이 있고 answer 가 있으면 통과(reward-hacking 방지)."""
    m = _FORMAT_RE.search(completion or '')
    if not m:
        return False
    return len(m.group(1).strip()) >= _MIN_THINK_CHARS


def aggregate(rubric: List[dict], verdicts: dict) -> float:
    """explicit RaR 집계: r = Σ wⱼ·cⱼ / Σ wⱼ. verdicts[key] ∈ {0,1}(없으면 0)."""
    num = 0.0
    den = 0.0
    for item in rubric:
        w = item['weight']
        den += w
        c = verdicts.get(item['key'])
        if c is None:
            continue
        num += w * (1.0 if float(c) >= 0.5 else 0.0)
    return (num / den) if den > 0 else 0.0


def build_judge_prompt(question: str, reference: str, answer_block: str,
                       rubric: List[dict]) -> str:
    lines = [
        "You are a strict grader for medical image VQA. Given the IMAGE, the QUESTION, a "
        "REFERENCE ANSWER (gold), and the MODEL OUTPUT (with <think> reasoning and <answer>), "
        "judge EACH criterion independently as 1 (satisfied) or 0 (not satisfied).",
        "",
        f"QUESTION: {question}",
        f"REFERENCE ANSWER: {reference}",
        f"MODEL OUTPUT: {answer_block}",
        "",
        "CRITERIA:",
    ]
    for item in rubric:
        lines.append(f"- {item['key']} ({item['title']}): {item['description']}")
    keys = ', '.join(f'"{it["key"]}": 0 or 1' for it in rubric)
    lines += [
        "",
        f"Respond with ONLY a JSON object: {{{keys}}}. No other text.",
    ]
    return '\n'.join(lines)


def parse_verdicts(text: str, rubric: List[dict]) -> Optional[dict]:
    m = _JSON_RE.search(text or '')
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for item in rubric:
        v = obj.get(item['key'])
        if v is None:
            return None
        try:
            out[item['key']] = 1.0 if float(v) >= 0.5 else 0.0
        except Exception:
            return None
    return out


def _image_to_data_url(path: str) -> Optional[str]:
    try:
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip('.').lower() or 'png'
        return f'data:image/{ext};base64,{b64}'
    except Exception:
        return None


class ClinicalJudgeReward(AsyncORM):
    """임상 멀티모달 루브릭 보상(비동기). 각 completion 을 RaR 루브릭으로 0~1 채점."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self._client = None  # lazy

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY,
                                       timeout=JUDGE_TIMEOUT)
        return self._client

    async def _score_one(self, completion, solution, image_path):
        # 형식 게이트: 위반 시 judge 호출 생략(비용↓), 0.0
        if not format_ok(completion):
            return 0.0
        reference = str(solution).strip()
        measurement = is_measurement(reference)
        rubric = build_rubric(reference, measurement)
        answer_block = completion.strip()
        prompt = build_judge_prompt(
            question='(see image)', reference=reference,
            answer_block=answer_block, rubric=rubric)

        content = [{'type': 'text', 'text': prompt}]
        if SEND_IMAGE and image_path:
            data_url = _image_to_data_url(image_path)
            if data_url:
                content.append({'type': 'image_url', 'image_url': {'url': data_url}})
        try:
            client = self._get_client()
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{'role': 'user', 'content': content}],
                temperature=0.0, max_tokens=JUDGE_MAX_TOKENS,
                extra_body=_EXTRA_BODY)
            text = _msg_text(resp)
        except Exception:
            return 0.0  # 타임아웃/네트워크/서버오류 → 학습 중단 방지
        verdicts = parse_verdicts(text, rubric)
        if verdicts is None:
            return 0.0  # 파싱 불가 → 0
        return aggregate(rubric, verdicts)

    async def __call__(self, completions, **kwargs) -> List[float]:
        solutions = kwargs.get('solution') or [''] * len(completions)
        images = kwargs.get('images') or [None] * len(completions)

        def first_img(x):
            if isinstance(x, (list, tuple)):
                return x[0] if x else None
            return x

        tasks = [
            self._score_one(c, solutions[i] if i < len(solutions) else '',
                            first_img(images[i]) if i < len(images) else None)
            for i, c in enumerate(completions)
        ]
        return list(await asyncio.gather(*tasks))


# ms-swift 가 --reward_funcs clinical_judge 로 찾을 수 있게 레지스트리에 등록
orms['clinical_judge'] = ClinicalJudgeReward
