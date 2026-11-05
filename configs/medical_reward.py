"""
medical_reward.py — [3단계] 의료 멀티모달 VQA 루브릭 보상 (Rubric-as-a-Reward / RaR)

ms-swift GRPO 의 --external_plugins 로 로드되어 'clinical_judge' 보상을 등록한다.
실행: swift rlhf ... --external_plugins configs/medical_reward.py \
                     --reward_funcs format_think clinical_judge

설계(상세 docs/2026-1차/30_stage3_medical_rl.html §4):
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
# 기본 judge = KHDP deepseek-v4-1-flash-ax (2026-10-15 judge_audit 승자). KISTI 계산노드는 외부망이 없어
# 30_medical_rl.slurm 이 자체 judge 서버 주소·모델을 env 로 덮는다.
JUDGE_BASE_URL = os.environ.get('JUDGE_BASE_URL', 'https://api.khdp.net/v1')
JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'deepseek-v4-1-flash-ax')
JUDGE_API_KEY = os.environ.get('JUDGE_API_KEY', 'EMPTY')
JUDGE_TIMEOUT = float(os.environ.get('JUDGE_TIMEOUT', '60'))
JUDGE_MAX_TOKENS = int(os.environ.get('JUDGE_MAX_TOKENS', '512'))
MEASURE_TOL_PCT = int(os.environ.get('JUDGE_MEASURE_TOL_PCT', '15'))
SEND_IMAGE = os.environ.get('JUDGE_SEND_IMAGE', '1') == '1'   # judge 가 멀티모달이면 1
# Qwen3 계열 judge 는 추론모델 → JSON 판정만 필요하므로 thinking 끔(빈 content/토큰낭비 방지).
JUDGE_THINK = os.environ.get('JUDGE_THINK', '0') == '1'
_EXTRA_BODY = {} if JUDGE_THINK else {'chat_template_kwargs': {'enable_thinking': False}}
# 스모크/디버그: swift 가 넘기는 kwargs(특히 images) 구조를 첫 호출에 1회 출력.
JUDGE_DEBUG = os.environ.get('JUDGE_DEBUG', '0') == '1'
_DEBUG_DONE = []
# 증거 인식 크레딧(MetaRubric, arXiv 2610.02824 의 Vacuous Credit 대응): judge 가 c2 의 근거 문장을
# 인용해야 하고, 그 인용이 <think> 에 실제로 있어야 c2 를 인정한다. c2 가 없으면 c4(과장 없음)도 무효.
# ⚠️ 기본 OFF — judge 가 인용 필드를 지키는지 GPU 에서 분포 프로브로 확인한 뒤 켤 것
# (안 지키면 전 응답 c2=0 으로 보상이 붕괴한다).
EVIDENCE_GATE = os.environ.get('RUBRIC_EVIDENCE', '0') == '1'
_MIN_EVIDENCE_CHARS = 8


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
         'description': ("Important Criteria: The <think> reasoning refers to an image-relevant observation "
                         "that is CONSISTENT with the actual image and supports the answer. It need not be "
                         "highly detailed, but it must NOT contradict or fabricate what the image shows "
                         "(e.g. describing structures or scenery absent from the image fails this).")},
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
                       rubric: List[dict], evidence: bool = False) -> str:
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
    if evidence:
        keys = '"c2_evidence": "<quote>", ' + keys   # 인용을 먼저 쓰게 해 c2 판정이 인용에 근거하도록
        lines += [
            "",
            "FIRST fill c2_evidence: copy VERBATIM from the <think> text one sentence that states an "
            "observation about the image (no braces or newlines). If <think> states no such observation, "
            "use an empty string. Then judge c2: c2 may be 1 ONLY if c2_evidence is non-empty.",
        ]
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


def parse_evidence(text: str) -> str:
    m = _JSON_RE.search(text or '')
    try:
        return str(json.loads(m.group(0)).get('c2_evidence') or '') if m else ''
    except Exception:
        return ''


def apply_evidence_gate(verdicts: dict, evidence: str, completion: str) -> dict:
    """c2 는 인용이 <think> 에 실재할 때만 인정, c2 가 무효면 c4 도 무효(아무 주장 안 한 응답이
    '과장 없음' 을 공짜로 받는 Vacuous Credit 차단). 공백·대소문자는 무시하고 비교."""
    out = dict(verdicts)
    if not evidence_ok(evidence, completion):
        out['c2'] = 0.0
    if not out.get('c2'):
        out['c4'] = 0.0
    return out


def evidence_ok(evidence: str, completion: str) -> bool:
    """judge 가 인용한 문장이 <think> 안에 실제로 있는가(공백·대소문자 무시, 8자 이상)."""
    norm = lambda s: ' '.join((s or '').split()).lower()
    m = _FORMAT_RE.search(completion or '')
    ev = norm(evidence)
    return bool(m) and len(ev) >= _MIN_EVIDENCE_CHARS and ev in norm(m.group(1))


# ---- 루브릭 조합 실험(2026-11-05) -------------------------------------------
# C0 = 위의 v1 그대로(기본). C1~C4 는 앞 단계에 하나씩 더한다 — 단계별 효과를 따로 보기 위해서다.
#   C1 질문 원문 전달(RaR 2507.17746 의 judge 입력) · C2 정답을 코드로(RLCF 2507.18624, RLR³ 2605.30244)
#   C3 시각근거를 원자 항목으로 분할 + 인용 실재 확인(V-Rubrics 2608.25580, 복합 기준 부분충족 해킹 2605.12474)
#   C4 '환각 없음' 가점 → 환각·과장 감점(HealthBench 2505.08775 의 음수 기준)
# 학습에서는 RUBRIC_CONFIG env 로 고른다. 오프라인 비교는 scripts/rubric_ablation.py.
_STEPS = ('question', 'code_c1', 'split_c2', 'penalties')
RUBRIC_CONFIGS = {f'C{i}': {s: j < i for j, s in enumerate(_STEPS)} for i in range(len(_STEPS) + 1)}
RUBRIC_CONFIG = os.environ.get('RUBRIC_CONFIG', 'C0')
assert RUBRIC_CONFIG in RUBRIC_CONFIGS, f'RUBRIC_CONFIG={RUBRIC_CONFIG} (가능: {sorted(RUBRIC_CONFIGS)})'
W_EVIDENCE, W_SUPPORT, W_P_HALLUC, W_P_OVERCLAIM = 1, 1, -4, -2


def build_rubric_cfg(reference: str, measurement: bool, cfg: dict) -> List[dict]:
    """조합별 루브릭. weight<0 은 감점 항목(판정 1 = 그 문제가 있음). c2a(인용 실재)는 judge 가 아니라 코드가 판정한다."""
    base = {it['key']: it for it in build_rubric(reference, measurement)}
    items = [base['c1']]
    if cfg['split_c2']:
        items += [
            {'key': 'c2b', 'title': 'Observation matches image', 'weight': W_IMPORTANT,
             'description': ("Important Criteria: The image observation quoted in c2_evidence is CONSISTENT with "
                             "what the image actually shows (not fabricated and not contradicted by the image).")},
            {'key': 'c2c', 'title': 'Observation supports answer', 'weight': W_SUPPORT,
             'description': "Optional Criteria: The quoted observation is actually used in the reasoning to reach the <answer>."}]
    else:
        items.append(base['c2'])
    if 'c3' in base:
        items.append(base['c3'])
    if cfg['penalties']:
        items += [
            {'key': 'p1', 'title': 'Hallucinated finding', 'weight': W_P_HALLUC,
             'description': "Pitfall (1 = PRESENT): The output asserts a finding, structure, or detail that is NOT present in the image."},
            {'key': 'p2', 'title': 'Overclaim beyond the question', 'weight': W_P_OVERCLAIM,
             'description': "Pitfall (1 = PRESENT): The <answer> adds diagnoses or claims that the QUESTION did not ask for."}]
    else:
        items.append(base['c4'])
    return items


def aggregate_cfg(items: List[dict], verdicts: dict, cfg: dict) -> float:
    """HealthBench 식: (충족한 가점 + 해당된 감점) / 가점 합, [0,1] 로 자른다. 판정이 없으면 0 으로 본다."""
    pos = sum(it['weight'] for it in items if it['weight'] > 0) + (W_EVIDENCE if cfg['split_c2'] else 0)
    got = sum(it['weight'] * verdicts.get(it['key'], 0.0) for it in items)
    if cfg['split_c2']:
        got += W_EVIDENCE * verdicts.get('c2a', 0.0)
    return min(1.0, max(0.0, got / pos)) if pos else 0.0


def code_correct(completion: str, reference: str) -> bool:
    """정답을 코드로 판정(accuracy.AccuracyMix, 참조답의 마지막 줄 — medix 는 첫 줄이 모달리티 태그다).
    판정 못 하는 환경(math_verify 없음 등)이면 False → judge 가 c1 을 본다."""
    try:
        import importlib.util as u
        spec = u.spec_from_file_location('accuracy', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'accuracy.py'))
        acc = u.module_from_spec(spec); spec.loader.exec_module(acc)
        gold = (reference or '').strip().splitlines()[-1] if (reference or '').strip() else ''
        return acc.AccuracyMix()([completion], [gold])[0] >= 0.5
    except Exception:
        return False


def build_judge_prompt_cfg(question: str, reference: str, answer_block: str, items: List[dict], evidence: bool) -> str:
    keys = ', '.join(f'"{it["key"]}": 0 or 1' for it in items)
    lines = [
        "You are a strict grader for medical image VQA. Given the IMAGE, the QUESTION, a REFERENCE ANSWER (gold), and "
        "the MODEL OUTPUT (with <think> reasoning and <answer>), judge EACH criterion independently as 0 or 1.",
        "For Pitfall criteria, 1 means the problem IS present in the output.", "",
        f"QUESTION: {question}", f"REFERENCE ANSWER: {reference}", f"MODEL OUTPUT: {answer_block}", "", "CRITERIA:",
    ] + [f"- {it['key']} ({it['title']}): {it['description']}" for it in items]
    if evidence:
        keys = '"c2_evidence": "<quote>", ' + keys
        lines += ["", "FIRST fill c2_evidence: copy VERBATIM from the <think> text one sentence that states an observation "
                      "about the image (no braces or newlines). If <think> states no such observation, use an empty string."]
    return '\n'.join(lines + ["", f"Respond with ONLY a JSON object: {{{keys}}}. No other text."])


async def score_cfg(cfg: dict, question: str, reference: str, completion: str, image_path, client, model: str) -> dict:
    """한 응답을 조합 cfg 로 채점한다. 반환 {'reward','v','err','tokens'} — 오프라인 실험과 학습이 같은 함수를 쓴다."""
    if not format_ok(completion):
        return {'reward': 0.0, 'v': {}, 'err': 'format', 'tokens': 0}
    reference = str(reference).strip()
    items = build_rubric_cfg(reference, is_measurement(reference), cfg)
    pre = {'c1': 1.0} if cfg['code_c1'] and code_correct(completion, reference) else {}
    jitems = [it for it in items if it['key'] not in pre]
    q = question if cfg['question'] and question else '(see image)'
    # 항목 구성이 v1 과 같은 C0~C2 는 v1 프롬프트를 그대로 쓴다 — 단계별 차이가 문구 변경과 섞이지 않게
    prompt = (build_judge_prompt_cfg(q, reference, completion.strip(), jitems, evidence=cfg['split_c2'])
              if cfg['split_c2'] or cfg['penalties'] else build_judge_prompt(q, reference, completion.strip(), jitems))
    content = [{'type': 'text', 'text': prompt}]
    if SEND_IMAGE and image_path:
        du = _image_to_data_url(image_path)
        if du:
            content.append({'type': 'image_url', 'image_url': {'url': du}})
    try:
        resp = await client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': content}],
                                                    temperature=0.0, max_tokens=JUDGE_MAX_TOKENS, extra_body=_EXTRA_BODY)
        text = _msg_text(resp)
        tokens = getattr(getattr(resp, 'usage', None), 'total_tokens', None)
    except Exception as e:
        return {'reward': 0.0, 'v': {}, 'err': f'api:{type(e).__name__}', 'tokens': 0}
    v = parse_verdicts(text, jitems)
    if v is None:
        return {'reward': 0.0, 'v': {}, 'err': 'parse', 'tokens': tokens}
    v = {**pre, **v}
    if cfg['split_c2']:
        v['c2a'] = 1.0 if evidence_ok(parse_evidence(text), completion) else 0.0
        if not v['c2a']:   # 인용이 없으면 그 인용에 대한 판정(c2b·c2c)도 무효
            v['c2b'] = v['c2c'] = 0.0
    return {'reward': aggregate_cfg(items, v, cfg), 'v': v, 'err': None, 'tokens': tokens}


def _image_to_data_url(img) -> Optional[str]:
    """이미지를 data URL 로. swift/GRPO 는 images 를 str 경로가 아니라
    HF datasets 포맷 dict {'bytes': <bytes|None>, 'path': <str|None>} 로 넘긴다(실측).
    str 경로 / 그 dict / PIL.Image 를 모두 지원."""
    try:
        raw = None
        ext = 'png'
        # HF datasets 이미지 dict: bytes 우선, 없으면 path
        if isinstance(img, dict):
            if img.get('bytes'):
                raw = img['bytes']
            elif img.get('path'):
                img = img['path']
            else:
                return None
        if raw is None and isinstance(img, str):
            ext = os.path.splitext(img)[1].lstrip('.').lower() or 'png'
            with open(img, 'rb') as f:
                raw = f.read()
        if raw is None and hasattr(img, 'save'):        # PIL.Image 폴백
            import io
            buf = io.BytesIO()
            img.convert('RGB').save(buf, format='PNG')
            raw = buf.getvalue()
        if raw is None:
            return None
        b64 = base64.b64encode(raw).decode()
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

    async def _score_one(self, completion, solution, image_path, question=''):
        if RUBRIC_CONFIG != 'C0':   # 조합 실험 경로. C0 은 아래 v1 경로를 그대로 탄다
            return (await score_cfg(RUBRIC_CONFIGS[RUBRIC_CONFIG], question, solution, completion, image_path,
                                    self._get_client(), JUDGE_MODEL))['reward']
        # 형식 게이트: 위반 시 judge 호출 생략(비용↓), 0.0
        if not format_ok(completion):
            return 0.0
        reference = str(solution).strip()
        measurement = is_measurement(reference)
        rubric = build_rubric(reference, measurement)
        answer_block = completion.strip()
        prompt = build_judge_prompt(
            question='(see image)', reference=reference,
            answer_block=answer_block, rubric=rubric, evidence=EVIDENCE_GATE)

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
        if EVIDENCE_GATE:
            verdicts = apply_evidence_gate(verdicts, parse_evidence(text), completion)
        return aggregate(rubric, verdicts)

    async def __call__(self, completions, **kwargs) -> List[float]:
        solutions = kwargs.get('solution') or [''] * len(completions)
        images = kwargs.get('images') or [None] * len(completions)

        def first_img(x):
            if isinstance(x, (list, tuple)):
                return x[0] if x else None
            return x

        if JUDGE_DEBUG and not _DEBUG_DONE:
            _DEBUG_DONE.append(1)
            im0 = first_img(images[0]) if images else None
            du = _image_to_data_url(im0) if im0 else None
            print('[JUDGE_DEBUG] kwargs keys =', sorted(kwargs.keys()), flush=True)
            print(f'[JUDGE_DEBUG] n_completions={len(completions)} '
                  f'solution[0]={str(solutions[0])[:60]!r}', flush=True)
            p0 = im0.get('path') if isinstance(im0, dict) else (im0 if isinstance(im0, str) else None)
            print(f'[JUDGE_DEBUG] images[0] type={type(images[0]).__name__} '
                  f'first_img type={type(im0).__name__} value={str(im0)[:120]!r}', flush=True)
            print(f'[JUDGE_DEBUG] resolved_path={p0!r} path_exists='
                  f'{os.path.isfile(p0) if isinstance(p0, str) else "N/A"} '
                  f'data_url_ok={bool(du)} '
                  f'(→ 시각근거 c2 채점에 이미지 {"전달됨" if du else "누락"})', flush=True)

        msgs = kwargs.get('messages') or [None] * len(completions)

        def question_of(m):   # 마지막 user 메시지의 글자 부분(<image> 자리표시 제거)
            users = [x for x in (m or []) if isinstance(x, dict) and x.get('role') == 'user']
            c = users[-1].get('content') if users else ''
            if isinstance(c, list):
                c = ' '.join(p.get('text', '') for p in c if isinstance(p, dict))
            return str(c or '').replace('<image>', '').strip()

        tasks = [
            self._score_one(c, solutions[i] if i < len(solutions) else '',
                            first_img(images[i]) if i < len(images) else None,
                            question_of(msgs[i]) if i < len(msgs) else '')
            for i, c in enumerate(completions)
        ]
        return list(await asyncio.gather(*tasks))


# ms-swift 가 --reward_funcs clinical_judge 로 찾을 수 있게 레지스트리에 등록
orms['clinical_judge'] = ClinicalJudgeReward
