#!/usr/bin/env python3
"""judge smoke client — 라이브 judge 서버에 실제 medical_reward 경로를 태워 검증.

컨테이너 내 실행, env: JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY.
실제 medix 1건으로 (좋은답 vs 나쁜답) 보상을 계산 → 단조성·멀티모달·파싱 확인.
또한 judge 원응답(raw)도 1건 출력해 파싱 디버깅 지원.
"""
import os, sys, json, asyncio, base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
import medical_reward as M  # noqa: E402

# --- 실제 medix 샘플 1건 ---
row = json.loads(open('work/data/medix_rl_train.jsonl').readline())
img = row['images'][0]
sol = str(row['solution']).strip()
q = row['messages'][0]['content']
ans_text = sol.splitlines()[-1].strip()  # 모달리티 태그 제외한 실제 답
print(f"[case] Q={q[:100]!r}\n       REF={sol!r}  img={os.path.basename(img)}  측정형={M.is_measurement(sol)}")

good = (f"<think>The CT image shows a well-circumscribed multicystic mass; I measured its "
        f"long and short axes on the axial slice.</think><answer>\\boxed{{{ans_text}}}</answer>")
bad = ("<think>There seems to be a huge mass, looks malignant and spreading everywhere.</think>"
       "<answer>\\boxed{a giant malignant tumor}</answer>")

# --- (A) raw judge 응답 1건 (파싱 디버깅용) ---
async def raw_probe():
    rubric = M.build_rubric(sol, M.is_measurement(sol))
    prompt = M.build_judge_prompt('(see image)', sol, good, rubric)
    content = [{'type': 'text', 'text': prompt}]
    du = M._image_to_data_url(img)
    if du:
        content.append({'type': 'image_url', 'image_url': {'url': du}})
    from openai import AsyncOpenAI
    cli = AsyncOpenAI(base_url=os.environ['JUDGE_BASE_URL'], api_key=os.environ.get('JUDGE_API_KEY', 'EMPTY'))
    r = await cli.chat.completions.create(model=os.environ['JUDGE_MODEL'],
                                          messages=[{'role': 'user', 'content': content}],
                                          temperature=0.0, max_tokens=M.JUDGE_MAX_TOKENS,
                                          extra_body=M._EXTRA_BODY)
    msg = r.choices[0].message
    print(f"\n[raw 진단] finish_reason={r.choices[0].finish_reason}")
    print(f"  content={getattr(msg,'content',None)!r}")
    print(f"  reasoning_content={getattr(msg,'reasoning_content',None)!r}")
    txt = M._msg_text(r)
    print(f"[파싱 결과] {M.parse_verdicts(txt, rubric)}")

# --- (B) 전체 보상 경로 (좋은답 vs 나쁜답) ---
async def reward_probe():
    rw = M.ClinicalJudgeReward()
    scores = await rw.__call__([good, bad], solution=[sol, sol], images=[[img], [img]])
    print(f"\n[보상] good={scores[0]:.3f}  bad={scores[1]:.3f}")
    ok = scores[0] > scores[1]
    print("RESULT:", "✅ PASS (good>bad)" if ok else "⚠️ CHECK (단조성 미성립)")
    return ok

async def main():
    try:
        await raw_probe()
    except Exception as e:
        print(f"[raw probe 오류] {type(e).__name__}: {e}")
    ok = await reward_probe()
    sys.exit(0 if ok else 2)

asyncio.run(main())
