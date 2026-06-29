#!/usr/bin/env python3
"""judge 분포 프로브 (spec §8) — 라이브 judge 에 medix N건 ×3변형 채점 → 캘리브레이션 점검.

변형:
  good   : 정답 + 근거있는 think       → 높게(특히 c1·c2=1)
  wrong  : 오답 + 근거있는 think       → 낮게(c1=0)  [정답 분리 검증]
  halluc : 정답 + 엉뚱한 영상묘사 think → c2=0 기대   [judge 가 이미지 실제로 보는지]
env: JUDGE_BASE_URL/MODEL/API_KEY, PROBE_N(기본100), PROBE_CONC(기본8).
"""
import os, sys, json, asyncio, statistics as st
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
import medical_reward as M  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

N = int(os.environ.get('PROBE_N', '100'))
CONC = int(os.environ.get('PROBE_CONC', '8'))
rows = []
for l in open('work/data/medix_rl_train.jsonl'):
    if len(rows) >= N:
        break
    d = json.loads(l)
    if d.get('images') and d.get('solution'):
        rows.append(d)
print(f"[probe] N={len(rows)} conc={CONC} model={os.environ['JUDGE_MODEL']}")

cli = AsyncOpenAI(base_url=os.environ['JUDGE_BASE_URL'], api_key=os.environ.get('JUDGE_API_KEY', 'EMPTY'))
sem = asyncio.Semaphore(CONC)

GROUNDED = ("On the provided image I localized the relevant region and examined its appearance "
            "(location, shape, margins, intensity) to derive the answer.")
ABSURD = "The image clearly shows a sunny tropical beach with palm trees, blue sky and ocean waves."


def variants(ref):
    ans = ref.splitlines()[-1].strip()
    return {
        'good':   f"<think>{GROUNDED}</think><answer>\\boxed{{{ans}}}</answer>",
        'wrong':  f"<think>{GROUNDED}</think><answer>\\boxed{{an unrelated and incorrect answer}}</answer>",
        'halluc': f"<think>{ABSURD}</think><answer>\\boxed{{{ans}}}</answer>",
    }


async def score(comp, ref, img):
    if not M.format_ok(comp):
        return 0.0, None
    rubric = M.build_rubric(ref, M.is_measurement(ref))
    prompt = M.build_judge_prompt('(see image)', ref, comp, rubric)
    content = [{'type': 'text', 'text': prompt}]
    du = M._image_to_data_url(img)
    if du:
        content.append({'type': 'image_url', 'image_url': {'url': du}})
    async with sem:
        try:
            r = await cli.chat.completions.create(
                model=os.environ['JUDGE_MODEL'], messages=[{'role': 'user', 'content': content}],
                temperature=0.0, max_tokens=M.JUDGE_MAX_TOKENS, extra_body=M._EXTRA_BODY)
            v = M.parse_verdicts(M._msg_text(r), rubric)
        except Exception:
            return 0.0, None
    if v is None:
        return 0.0, None
    return M.aggregate(rubric, v), v


async def main():
    tasks, meta = [], []
    for d in rows:
        ref = str(d['solution']).strip(); img = d['images'][0]
        for name, comp in variants(ref).items():
            tasks.append(score(comp, ref, img)); meta.append(name)
    res = await asyncio.gather(*tasks)

    agg = defaultdict(list); crit = defaultdict(lambda: defaultdict(list)); pf = defaultdict(int)
    bysample = defaultdict(dict)
    for i, (name, (sc, v)) in enumerate(zip(meta, res)):
        agg[name].append(sc); bysample[i // 3][name] = sc
        if v is None:
            pf[name] += 1
        else:
            for k, val in v.items():
                crit[name][k].append(val)

    print("\n===== 분포 프로브 결과 =====")
    for name in ['good', 'wrong', 'halluc']:
        s = agg[name]
        cr = {k: round(st.mean(vv), 2) for k, vv in sorted(crit[name].items())}
        print(f"[{name:6}] reward mean={st.mean(s):.3f} median={st.median(s):.3f} min={min(s):.2f} max={max(s):.2f} "
              f"parsefail={pf[name]}/{len(s)}")
        print(f"          criterion sat-rate: {cr}")

    mono = sum(1 for d in bysample.values() if d.get('good', 0) > d.get('wrong', 1)) / len(bysample)
    c2g = st.mean(crit['good']['c2']) if crit['good']['c2'] else float('nan')
    c2h = st.mean(crit['halluc']['c2']) if crit['halluc']['c2'] else float('nan')
    print(f"\n[판정] 단조성 good>wrong: {mono*100:.0f}%")
    print(f"[판정] c2 변별: good={c2g:.2f} vs halluc={c2h:.2f} (Δ={c2g-c2h:+.2f}) — 클수록 judge 가 이미지 실제로 봄")
    print(f"[판정] good 평균 {st.mean(agg['good']):.2f} / wrong {st.mean(agg['wrong']):.2f} / halluc {st.mean(agg['halluc']):.2f}")
    ok = (mono > 0.8) and (c2g - c2h > 0.2)
    print("RESULT:", "✅ 캘리브레이션 양호" if ok else "⚠️ 점검 필요(단조성 or c2 변별 약함)")

asyncio.run(main())
