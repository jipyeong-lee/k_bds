"""
build_coldstart_sft.py — VLAA-Thinking(clevr_math) → format cold-start SFT jsonl

목적: GRPO 직전 '형식 cold-start'. 모델에게 (1) <think>간결추론</think><answer>답</answer>
  출력 구조와 (2) "장황하게 끌지 않고 결론 내는" 습관을 주입한다.
  파일럿2에서 Format=0(77% truncation) 원인이 바로 이 형식/결론 부재였음.

데이터: UCSC-VLAA/VLAA-Thinking 의 clevr_math 서브셋(시각-수학/카운팅, ~5.9K).
  answer 필드가 이미 <think>...</think><answer>X</answer> 형식 → 거의 그대로 사용.
  ※ \boxed{} 는 사용하지 않음(컨벤션 결정). math_verify 가 평문 숫자답을 정상 검증함.

출력: ms-swift 표준 SFT jsonl ({messages:[system,user,assistant], images:[abspath]}).

사용 (컨테이너로 실행 — huggingface_hub 캐시 접근):
  singularity exec --bind <work> --env HF_HOME=...,HF_HUB_OFFLINE=1 <sandbox> \
    python scripts/build_coldstart_sft.py --out-dir work/data --val-ratio 0.02
"""
import argparse
import json
import os
import re

# 00_common.sh 의 SYSTEM_PROMPT 와 동일(컨벤션 통일). \boxed 미사용 버전.
DEFAULT_SYS = (
    "You are a multimodal reasoning assistant. Carefully examine the image(s) and "
    "reason step by step INSIDE <think> </think>, keeping the reasoning concise. "
    "Then give ONLY the final answer INSIDE <answer> </answer>. "
    "For multiple-choice, put only the letter, e.g. <answer>A</answer>."
)
IMG_ROOT = "/home01/k266a01/kbds_project/work/data/images/vlaa_clevr"

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANS_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def clean_answer(raw_answer, max_ans_chars):
    """VLAA answer 필드에서 <think>/<answer> 추출, 결론이 간결한 것만 통과."""
    mt = THINK_RE.search(raw_answer)
    ma = ANS_RE.search(raw_answer)
    if not mt or not ma:
        return None
    think = mt.group(1).strip()
    ans = ma.group(1).strip()
    if not think or not ans:
        return None
    # answer 안에 추론이 새어든(장황한) 케이스 제외 → 깔끔한 짧은 최종답만
    if len(ans) > max_ans_chars or "\n" in ans:
        return None
    return f"<think>\n{think}\n</think>\n<answer>{ans}</answer>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--prefix', default='clevr_math',
                    help='VLAA image prefix 필터(서브셋 선택)')
    ap.add_argument('--val-ratio', type=float, default=0.02)
    ap.add_argument('--limit', type=int, default=0, help='최대 레코드 수(0=전체)')
    ap.add_argument('--max-ans-chars', type=int, default=24)
    ap.add_argument('--system', default=DEFAULT_SYS)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from huggingface_hub import hf_hub_download
    src = hf_hub_download("UCSC-VLAA/VLAA-Thinking",
                          "VLAA-Thinking-SFT-126K.json", repo_type="dataset")

    train_p = os.path.join(args.out_dir, 'sft_coldstart_train.jsonl')
    val_p = os.path.join(args.out_dir, 'sft_coldstart_val.jsonl')
    n_tr = n_va = n_skip_fmt = n_skip_img = 0
    val_every = max(2, int(round(1 / args.val_ratio))) if args.val_ratio > 0 else 0

    with open(src, encoding='utf-8') as f, \
         open(train_p, 'w', encoding='utf-8') as ftr, \
         open(val_p, 'w', encoding='utf-8') as fva:
        kept = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img = rec.get('image')
            if not isinstance(img, str) or not img.startswith(args.prefix + '/'):
                continue
            assistant = clean_answer(rec.get('answer', ''), args.max_ans_chars)
            if assistant is None:
                n_skip_fmt += 1
                continue
            img_abs = os.path.join(IMG_ROOT, img)
            if not os.path.exists(img_abs):
                n_skip_img += 1
                continue
            question = rec['question'].strip()
            # ms-swift 멀티모달: user content 에 <image> 플레이스홀더
            user_content = "<image>\n" + question
            out = {
                'messages': [
                    {'role': 'system', 'content': args.system},
                    {'role': 'user', 'content': user_content},
                    {'role': 'assistant', 'content': assistant},
                ],
                'images': [img_abs],
            }
            is_val = val_every and (kept % val_every == 0)
            (fva if is_val else ftr).write(json.dumps(out, ensure_ascii=False) + '\n')
            if is_val:
                n_va += 1
            else:
                n_tr += 1
            kept += 1
            if args.limit and kept >= args.limit:
                break

    print(f"✅ cold-start SFT: train={n_tr} ({train_p}), val={n_va} ({val_p})")
    print(f"   skip(형식불량)={n_skip_fmt}, skip(이미지없음)={n_skip_img}")


if __name__ == '__main__':
    main()
