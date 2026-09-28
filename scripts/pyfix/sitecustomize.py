"""PyTorch 2.6+ 가 torch.load 기본값을 weights_only=True 로 바꿔 ms-swift/HF Trainer 의
rng_state_*.pth(자체 학습이 만든 신뢰 가능한 체크포인트) 재개 로드가 깨진다
(UnpicklingError: numpy._core.multiarray._reconstruct). 우리 체크포인트만 다루므로
기본값을 원복한다. runc.sh 가 PYTHONPATH 맨 앞에 이 디렉터리를 넣어 모든 파이썬 실행에 적용됨.
"""
import torch

_orig_load = torch.load


def _load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _load
