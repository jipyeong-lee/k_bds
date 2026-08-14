"""
download_dataset.py — HF 데이터셋을 프로젝트 HF 캐시로 사전 다운로드 (로그인 노드, 컨테이너 내 실행)
  컴퓨트 노드는 오프라인이므로 학습/변환 전에 받아 둔다.

사용:
  singularity exec --bind <work> --env HF_HUB_ENABLE_HF_TRANSFER=0 <sandbox> \
    python scripts/download_dataset.py <dataset_id>
  예) python scripts/download_dataset.py skylenage-ai/DeepVision-103K
      python scripts/download_dataset.py MBZUAI/medix-rl-data
  게이트 데이터면 export HF_TOKEN=hf_xxx (토큰은 ~/model_download.py 에 보유)
"""
import os
import sys
import time

os.environ.setdefault('HF_HOME', '/home01/k266a01/kbds_project/work/hf_cache')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')   # 컨테이너에 hf_transfer 없음

from huggingface_hub import snapshot_download

DATASET_ID = sys.argv[1] if len(sys.argv) > 1 else 'skylenage-ai/DeepVision-103K'
TOKEN = os.environ.get('HF_TOKEN')

print(f"Downloading dataset {DATASET_ID} -> HF_HOME={os.environ['HF_HOME']}")
for attempt in range(10):
    try:
        path = snapshot_download(repo_id=DATASET_ID, repo_type='dataset', token=TOKEN)
        print(f'✅ Download complete: {path}')
        break
    except Exception as e:
        delay = min(5 * (2 ** attempt), 120)
        print(f"[{attempt + 1}/10] Failed: {e}")
        if attempt < 9:
            print(f"Retrying in {delay}s..."); time.sleep(delay)
        else:
            print("❌ Max retries reached."); raise
