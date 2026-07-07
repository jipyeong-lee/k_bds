#!/usr/bin/env python3
"""run_healthbench.py — evalscope HealthBench 평가 (자체 vLLM 서버 2개: base + judge).

평가대상(base)·채점자(judge) 둘 다 로컬 vLLM OpenAI 엔드포인트로 서빙된 상태에서 호출.
env:
  HB_MODEL_ID(base9b) / HB_API_URL(http://localhost:8000/v1/) — 평가대상
  HB_JUDGE_ID(qwen36-judge) / HB_JUDGE_URL(http://localhost:8100/v1/) — 채점자
  HB_VERSION(Hard) / HB_LIMIT(20) / HB_MAXTOK(2048) / HB_WORKDIR
채점자는 오프라인이라 GPT-4.1 아님(Qwen3.6-27B) → 절대점수는 공식 리더보드와 직접비교 불가(내부 상대비교용).
"""
import os
import re
import json as _json
from evalscope import TaskConfig, run_task

# ── HealthBench 채점 견고화 monkey-patch ──────────────────────────────────
# 문제: 어댑터는 루브릭 항목마다 judge JSON을 파싱, criteria_met(bool)이 없으면
#   3회 재시도 후 ValueError → 전체 평가 abort. self-hosted judge(Qwen3.6)가
#   가끔 형식을 어기면 한 항목 때문에 전멸. → 파서가 항상 bool criteria_met을
#   담은 dict를 반환하게 해 개별 실패를 '미충족(False)'로 보수적 처리(전멸 방지).
import evalscope.benchmarks.healthbench.utils as _hb_utils  # noqa: E402


def _robust_parse_json_to_dict(s: str) -> dict:
    s = (s or '').strip()
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.S)          # thinking 제거(안전망)
    s = re.sub(r'^```(?:json)?\s*|\s*```$', '', s.strip())         # 코드펜스 제거
    d = {}
    try:
        d = _json.loads(s)
    except Exception:
        m = re.search(r'\{.*\}', s, re.S)                          # 첫 {...} 블록 추출
        if m:
            try:
                d = _json.loads(m.group(0))
            except Exception:
                d = {}
    if not isinstance(d, dict):
        d = {}
    cm = d.get('criteria_met', None)
    genuine_fail = False
    if isinstance(cm, str):
        d['criteria_met'] = cm.strip().lower() in ('true', 'yes', '1')
    elif not isinstance(cm, bool):
        m = re.search(r'criteria_met["\s:]+(true|false)', s, re.I)  # 최후: 원문 정규식
        d['criteria_met'] = bool(m and m.group(1).lower() == 'true')
        genuine_fail = m is None   # bool도 문자열도 정규식도 실패 = 진짜 파싱불가
    # 마커 분리: 진짜 파싱실패만 명시 태그(설명 단순 누락과 구분 → 확장실행 실패율 클린)
    if genuine_fail:
        d['explanation'] = '(robust-parse-FAILED: judge 형식오류→미충족 처리)'
    else:
        d.setdefault('explanation', '(no-explanation)')
    return d


_hb_utils.parse_json_to_dict = _robust_parse_json_to_dict
# ──────────────────────────────────────────────────────────────────────────

VERSION = os.environ.get('HB_VERSION', 'Hard')
_lim = int(os.environ.get('HB_LIMIT', '20'))
LIMIT = None if _lim <= 0 else _lim   # HB_LIMIT<=0 → 전량(서브셋 제한 없음)
WORKDIR = os.environ.get('HB_WORKDIR', 'work/eval_healthbench')
# 오프라인: 미리 받아둔 로컬 스냅샷 경로(존재하면 modelscope 다운로드 우회)
LOCAL_PATH = os.environ.get('HB_LOCAL_PATH', '')

hb_args = {'extra_params': {'version': VERSION}}
if LOCAL_PATH:
    hb_args['local_path'] = LOCAL_PATH

cfg = TaskConfig(
    model=os.environ.get('HB_MODEL_ID', 'base9b'),
    api_url=os.environ.get('HB_API_URL', 'http://localhost:8000/v1/'),
    api_key='EMPTY',
    eval_type='openai_api',
    datasets=['health_bench'],
    dataset_args={'health_bench': hb_args},
    limit=LIMIT,
    eval_batch_size=int(os.environ.get('HB_CONC', '8')),
    generation_config={
        'temperature': 0.0,
        'max_tokens': int(os.environ.get('HB_MAXTOK', '2048')),
    },
    # 채점자(LLM judge) = 자체 Qwen3.6-27B vLLM
    judge_strategy='auto',
    judge_worker_num=int(os.environ.get('HB_JUDGE_CONC', '8')),
    judge_model_args={
        'model_id': os.environ.get('HB_JUDGE_ID', 'qwen36-judge'),
        'api_url': os.environ.get('HB_JUDGE_URL', 'http://localhost:8100/v1/'),
        'api_key': 'EMPTY',
        # Qwen3.6은 thinking 모델 → 그대로면 content가 비어 JSON 파싱 실패.
        # vLLM chat_template_kwargs 로 thinking 끄고 JSON만 바로 뱉게.
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 1024,
            'extra_body': {'chat_template_kwargs': {'enable_thinking': False}},
        },
    },
    work_dir=WORKDIR,
)

print(f'[healthbench] version={VERSION} limit={LIMIT} model={cfg.model} judge={cfg.judge_model_args["model_id"]}')
res = run_task(task_cfg=cfg)
print('[healthbench] DONE')
print(res)
