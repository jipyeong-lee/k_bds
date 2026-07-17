"""
build_sft.py — RL jsonl(convert_to_swift 출력) → SFT 학습용 jsonl (cold-start 형식 정렬)

목적: GRPO 이전 1단계 SFT. 모델에게 "추론 후 \\boxed{} 최종답" 출력 형식을 학습시킨다.
  ⚠️ 한계: DeepVision/medix 는 gold CoT(추론 체인)가 없고 정답만 있으므로, 이 SFT 는
     '형식/정답 정렬'이다. 본격 추론 cold-start 가 필요하면 CoT 데이터셋
     (예: UCSC-VLAA/VLAA-Thinking) 을 convert 후 style=cot 로 추가 권장.

입력: convert_to_swift 가 만든 RL jsonl ({messages:[user], images, solution}).
출력: SFT jsonl ({messages:[system,user,assistant], images}) + train/val 분할.

사용 (컨테이너 불필요, 순수 파이썬):
  python scripts/build_sft.py \
     --input work/data/deepvision103k_train.jsonl:boxed \
     [--input work/data/medix_rl_train.jsonl:freeform] \
     --out-dir work/data --val-ratio 0.02 [--limit-per-input N]
"""
import argparse
import json
import os

SYS_BOXED = ("You are a multimodal reasoning assistant. You receive images and texts, perform "
             "step-by-step reasoning (including re-checking the image) before producing the final "
             "answer. Provide a clear, concise final answer inside a \\boxed{} tag. For multiple "
             "choice, put only the letter like \\boxed{A}.")
SYS_MED = ("You are a medical multimodal reasoning assistant. Carefully analyze the image and "
           "question, reason step by step about the clinical findings, then give a concise, "
           "clinically accurate answer.")


def assistant_target(solution, style):
    sol = str(solution).strip()
    if style == 'boxed':
        return f"\\boxed{{{sol}}}"
    return sol            # freeform / cot: 정답(또는 CoT 포함 참조답) 그대로


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', action='append', required=True,
                    help='RL jsonl 경로:스타일 (스타일=boxed|freeform). 여러 번 지정 가능.')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--val-ratio', type=float, default=0.02)
    ap.add_argument('--limit-per-input', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_p = os.path.join(args.out_dir, 'sft_train.jsonl')
    val_p = os.path.join(args.out_dir, 'sft_val.jsonl')
    n_train = n_val = 0
    with open(train_p, 'w', encoding='utf-8') as ftr, open(val_p, 'w', encoding='utf-8') as fva:
        for spec in args.input:
            path, _, style = spec.partition(':')
            style = style or 'boxed'
            sysmsg = SYS_MED if style in ('freeform', 'cot') and 'medix' in path.lower() else \
                     (SYS_MED if style == 'freeform' else SYS_BOXED)
            cnt = 0
            with open(path, encoding='utf-8') as f:
                for k, line in enumerate(f):
                    if args.limit_per_input and cnt >= args.limit_per_input:
                        break
                    rec = json.loads(line)
                    user = next((m for m in rec['messages'] if m['role'] == 'user'), None)
                    if not user or not rec.get('solution'):
                        continue
                    out = {'messages': [{'role': 'system', 'content': sysmsg},
                                        {'role': 'user', 'content': user['content']},
                                        {'role': 'assistant',
                                         'content': assistant_target(rec['solution'], style)}],
                           'images': rec.get('images', [])}
                    # 간단 결정적 분할(매 1/val_ratio 번째를 val 로)
                    is_val = args.val_ratio > 0 and (k % max(2, int(round(1 / args.val_ratio))) == 0)
                    (fva if is_val else ftr).write(json.dumps(out, ensure_ascii=False) + '\n')
                    if is_val:
                        n_val += 1
                    else:
                        n_train += 1
                    cnt += 1
            print(f'  {path} (style={style}): {cnt} 처리')
    print(f'✅ SFT train={n_train} ({train_p}), val={n_val} ({val_p})')


if __name__ == '__main__':
    main()
