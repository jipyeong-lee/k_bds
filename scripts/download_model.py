"""
download_model.py — 베이스 모델을 프로젝트 HF 캐시로 사전 다운로드 (로그인 노드에서 실행)
  컴퓨트 노드는 오프라인이므로 학습 전에 여기서 받아 둔다.
  ~/model_download.py 의 패턴(snapshot_download + 재시도)을 따르되:
    - 다운로드 위치를 프로젝트 HF_HOME 으로 (컨테이너 학습이 보는 경로와 동일)
    - 토큰은 환경변수 HF_TOKEN 으로 (평문 하드코딩 회피)

사용:
  export HF_TOKEN=hf_xxx          # ~/model_download.py 에 있는 토큰 재사용 가능
  python scripts/download_model.py [model_id]
  (게이트 모델은 사전에 HF 모델 페이지에서 라이선스 동의 필요)
"""
import os
import sys
import time

# 컨테이너 학습이 참조하는 캐시와 동일 위치 (00_common.sh 의 HF_HOME 과 일치)
os.environ.setdefault('HF_HOME', '/home01/k252a02/kbds_project/work/hf_cache')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')   # 고속 전송(hf_transfer 필요)
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from huggingface_hub import snapshot_download

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else 'google/gemma-4-12B-it'
TOKEN = os.environ.get('HF_TOKEN')   # 없으면 캐시된 로그인 사용

print(f"Downloading {MODEL_ID} -> HF_HOME={os.environ['HF_HOME']}")
max_retries = 10
for attempt in range(max_retries):
    try:
        path = snapshot_download(repo_id=MODEL_ID, token=TOKEN)
        print(f'✅ Download complete: {path}')
        break
    except Exception as e:
        delay = min(5 * (2 ** attempt), 120)
        print(f"[{attempt + 1}/{max_retries}] Failed: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {delay}s...")
            time.sleep(delay)
        else:
            print("❌ Max retries reached.")
            raise
