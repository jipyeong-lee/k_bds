"""
medical_reward.py — [3단계] 의료 특화 개방형 RL 복합 보상 plugin (ms-swift 4.x)

ms-swift GRPO 의 --external_plugins 로 로드되어 'clinical_judge' 보상을 등록한다.
실행: swift rlhf ... --external_plugins configs/medical_reward.py \
                     --reward_funcs format clinical_judge

LLM-as-a-judge 는 vLLM judge 서버(OpenAI 호환)로 API 호출 → I/O 바운드이므로
AsyncORM 을 사용(asyncio.gather 로 배치 병렬 채점).

실제 구현 시 TODO:
  - judge 모델을 별도 vLLM 서버로 기동(OpenAI 호환 endpoint), base_url 연결
  - 임상 루브릭(정확성/안전성/근거제시) 프롬프트 설계 → 0.0~1.0 스칼라 파싱
"""
from typing import List

from swift.rewards import AsyncORM, orms


class ClinicalJudgeReward(AsyncORM):
    """임상 유효성 LLM 심판 보상(비동기). completions 각 항목을 0.0~1.0 로 채점."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        # TODO: judge 클라이언트 초기화 (예: vLLM OpenAI 호환 endpoint)
        # from openai import AsyncOpenAI
        # self.client = AsyncOpenAI(base_url='http://127.0.0.1:8100/v1', api_key='EMPTY')
        # self.judge_model = 'Qwen/Qwen2.5-VL-7B-Instruct'

    async def __call__(self, completions: List[str], **kwargs) -> List[float]:
        # kwargs 에 solution/messages 등 데이터셋 컬럼이 전달됨
        import asyncio

        async def score_one(completion, idx):
            # --- TODO: 실제 LLM 심판 호출로 교체 ---
            # prompt = build_clinical_rubric(completion, kwargs, idx)
            # resp = await self.client.chat.completions.create(model=self.judge_model, ...)
            # return parse_score_0_1(resp)
            return 0.0

        return list(await asyncio.gather(*[score_one(c, i) for i, c in enumerate(completions)]))


# ms-swift 가 --reward_funcs clinical_judge 로 찾을 수 있게 레지스트리에 등록
orms['clinical_judge'] = ClinicalJudgeReward
