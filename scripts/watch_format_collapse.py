#!/usr/bin/env python3
"""watch_format_collapse.py — GRPO 학습 로그를 폴링해 형식 붕괴를 조기에 잡고 잡을 세운다.

배경: job 73924/73925 는 step ~900 에서 형식이 무너진 뒤 **13시간 35분**을 더 돌았다
      (109 GPU-h 낭비). 붕괴를 "보고 있던 장치"가 아예 없었기 때문이다.
      상세 = docs/stage2_run73924_postmortem.md §1-5.

사용:
    # 실전 — 학습 잡 안에서 백그라운드로
    python3 scripts/watch_format_collapse.py --log logs/grpo_adv_12345.log \
        --job-id 12345 --stop-step 960 --verdict logs/verdict_12345.json

    # 임계값 검증 — 과거 로그를 live 처럼 재생해 "언제 울렸을지" 계산 (GPU 불필요)
    python3 scripts/watch_format_collapse.py --simulate \
        --log logs/grpo_adv_73924.log logs/grpo_adv_73925.log --verdict /dev/stdout

  * 표준 라이브러리만 쓴다 — 호스트 python3 로 돈다(컨테이너/로더 진입 불필요).
  * --job-id 를 주면 정지 조건 충족 시 `scancel`. 빼면 관측만 한다.

정지 조건 (먼저 걸리는 것):
  1) COLLAPSE — FormatThink 최근 W step 평균이 임계 미만
  2) DONE     — global_step 이 --stop-step 도달 (재현 실험용 정상 종료)

경보(정지 안 함, 기록만):
  3) KL_WARN  — kl 최근 창 중앙값이 기준선의 --kl-ratio 배 초과.
     73924 에서 이것이 형식 지표보다 먼저 움직였다(§1-5-1). 이번에도 정말 선행하는지
     재려고 기록만 하고 세우지는 않는다. verdict 의 kl_lead_steps 가 그 답이다.

     ⚠️ 절대 임계로 걸면 안 된다 — kl 은 정상 런에서도 단조 상승한다(73924: 6e-4 → 3e-2).
        그래서 "기준선 대비 배수" 로 판정한다.
     ⚠️ clipped_ratio 가 높으면 kl 은 과소보고된다. overlong_filter 가 절단 롤아웃을
        completion_mask 에서 빼는 코드가 KL 계산보다 **앞줄**이라(grpo_trainer.py:1132),
        남은 토큰에서만 KL 이 측정된다. 그래서 verdict 에 clipped_ratio 를 같이 남긴다.
"""
import argparse
import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

BLOCK = re.compile(r"\{'loss'.*?\}")
FMT = "rewards/FormatThink/mean"
KL = "kl"
CLIP = "completions/clipped_ratio"
SNAP = (FMT, KL, CLIP, "completions/mean_length", "grad_norm", "reward")


def parse(paths, seen):
    """로그들을 처음부터 다시 읽어 step -> row 를 채운다.

    tail 이 아니라 전량 재파싱인 이유: 학습 로그는 수 MB 수준이고 폴링 간격이 분 단위라
    비용이 무시할 만하다. 반면 tail 은 부분 기록된 블록·재시작·NFS 지연에 취약하다.
    """
    for path in paths:
        try:
            raw = Path(path).read_text(errors="replace")
        except FileNotFoundError:
            continue
        for m in BLOCK.finditer(raw):
            try:
                d = ast.literal_eval(m.group(0))
            except Exception:
                continue
            try:
                step = int(str(d.get("global_step/max_steps", "")).split("/")[0])
            except ValueError:
                continue
            row = {}
            for k, v in d.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
            seen[step] = row          # 나중 파일이 이김(resume 구간)


def med(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def win(seen, key, lo, hi):
    """step 이 [lo, hi] 인 구간의 key 값들 (결측·NaN 은 건너뛴다)."""
    return [seen[s][key] for s in range(lo, hi + 1)
            if s in seen and seen[s].get(key) == seen[s].get(key) and key in seen[s]]


class Monitor:
    """seen(step->row) 위에서 판정만 하는 상태기계. live 와 simulate 가 이걸 공유한다."""

    def __init__(self, a):
        self.a = a
        self.start = None
        self.kl_base = a.kl_baseline_value or None
        self.kl_warn = None

    def step(self, seen, cur):
        """cur 까지 관측했다고 보고 (event, detail) 을 낸다. event 없으면 (None, info)."""
        a = self.a
        if self.start is None:
            self.start = min(seen)

        # KL 경보 — 기록만. 기준선은 **이동**이다(현재 창 바로 앞 B step 의 중앙값).
        #  고정 기준선은 못 쓴다: kl 은 정상 런에서도 단조 상승해서(73924 는 6e-4 → 3e-2)
        #  런 초반으로 기준을 잡으면 아무 일 없는 step 61 에서 울린다(실측으로 확인).
        #  --kl-baseline-value 를 주면 그 값으로 고정한다 — 재개 실행처럼 앞 이력이
        #  없는 경우에 쓴다(73924 의 step 700~849 평탄값 = 0.031).
        #  기준선 창이 **평탄할 때만** 경보를 무장한다. 학습 초반의 kl 은 정상적으로
        #  가파르게 오르므로(73924: step 1~150 에 6e-4 → 5e-3), 무장 조건이 없으면
        #  아무 일 없는 step 50 에서 울린다 — 실측으로 확인하고 넣은 조건이다.
        if self.kl_warn is None:
            kw = win(seen, KL, cur - a.kl_window + 1, cur)
            base, lo = None, cur - a.kl_window - a.kl_baseline + 1
            if a.kl_baseline_value:
                base = a.kl_baseline_value                    # 재개 실행 — 앞 이력이 없다
            else:
                bv = win(seen, KL, lo, cur - a.kl_window)
                if len(bv) >= a.kl_baseline:                  # 창이 꽉 차야 한다
                    h = len(bv) // 2
                    e, l = med(bv[:h]), med(bv[h:])
                    if e > 0 and l / e <= a.kl_stable:        # 평탄 → 무장
                        base = med(bv)
            if base and len(kw) >= a.kl_window // 2 and med(kw) > base * a.kl_ratio:
                self.kl_base, self.kl_warn = round(base, 6), cur

        fw = win(seen, FMT, cur - a.fmt_window + 1, cur)
        fmt = sum(fw) / len(fw) if fw else float("nan")
        if len(fw) >= a.fmt_min_samples and fmt < a.fmt_threshold:
            return "COLLAPSE", f"FormatThink {a.fmt_window}-step 평균 {fmt:.3f} < {a.fmt_threshold}"
        if a.stop_step and cur >= a.stop_step:
            return "DONE", f"stop_step {a.stop_step} 도달 — 붕괴 없음"
        return None, fmt

    def verdict(self, seen, outcome, step, note):
        return {
            "outcome": outcome, "step": step, "note": note,
            "start_step": self.start,
            "kl_baseline": self.kl_base,
            "kl_warn_step": self.kl_warn,
            "kl_lead_steps": (step - self.kl_warn
                              if self.kl_warn is not None and outcome == "COLLAPSE" else None),
            "at_stop": {k: seen.get(step, {}).get(k) for k in SNAP},
        }


def write_verdict(path, obj):
    if path != "/dev/stdout":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def simulate(a):
    """과거 로그를 step 순서대로 흘려보내며 "언제 울렸을지" 를 계산한다.

    임계값을 추측으로 정하지 않기 위한 장치다. GPU 를 쓰지 않는다.
    """
    full = {}
    parse(a.log, full)
    if not full:
        sys.exit("지표 블록을 하나도 파싱하지 못했다 — 로그 경로 확인")
    mon, seen = Monitor(a), {}
    for s in sorted(full):
        seen[s] = full[s]
        ev, detail = mon.step(seen, s)
        if ev:
            v = mon.verdict(seen, ev, s, detail)
            v["mode"] = "simulate"
            write_verdict(a.verdict, v)
            lead = v["kl_lead_steps"]
            print(f"[sim] {ev} @ step {s} — {detail}", file=sys.stderr)
            print(f"[sim] kl 기준선 {v['kl_baseline']} · KL_WARN @ {v['kl_warn_step']}"
                  + (f" → 형식보다 {lead} step 선행" if lead else ""), file=sys.stderr)
            return
    v = mon.verdict(seen, "NO_EVENT", max(seen), "로그 끝까지 조건 미충족")
    v["mode"] = "simulate"
    write_verdict(a.verdict, v)
    print(f"[sim] NO_EVENT — 마지막 step {max(seen)}", file=sys.stderr)


def live(a):
    seen, t0 = {}, time.time()
    mon = Monitor(a)
    print(f"[watch] log={a.log} stop_step={a.stop_step} fmt<{a.fmt_threshold} "
          f"job={a.job_id or '(관측만)'}", flush=True)
    while True:
        parse(a.log, seen)
        if seen:
            cur = max(seen)
            prev_warn = mon.kl_warn
            ev, detail = mon.step(seen, cur)
            if mon.kl_warn and not prev_warn:
                print(f"[watch] ⚠️ KL_WARN @ {mon.kl_warn} "
                      f"(기준선 {mon.kl_base:.5f} 의 {a.kl_ratio}배 초과)", flush=True)
            if ev:
                v = mon.verdict(seen, ev, cur, detail)
                v["mode"] = "live"
                v["elapsed_sec"] = round(time.time() - t0)
                write_verdict(a.verdict, v)
                print(f"[watch] {ev} @ step {cur} — {detail}", flush=True)
                print(f"[watch] verdict: {a.verdict}", flush=True)
                if a.job_id:
                    print(f"[watch] scancel {a.job_id}", flush=True)
                    subprocess.run(["scancel", a.job_id], check=False)
                return
            if cur % 10 == 0 and isinstance(detail, float) and detail == detail:
                print(f"[watch] step {cur}  fmt={detail:.3f}", flush=True)
        if a.timeout and time.time() - t0 > a.timeout:
            cur = max(seen) if seen else -1
            v = mon.verdict(seen, "TIMEOUT", cur, f"{a.timeout}s 경과")
            v["mode"] = "live"
            write_verdict(a.verdict, v)
            print(f"[watch] TIMEOUT @ step {cur}", flush=True)
            return
        time.sleep(a.poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", nargs="+", required=True)
    ap.add_argument("--verdict", required=True, help="판정 JSON 출력 경로 (/dev/stdout 가능)")
    ap.add_argument("--job-id", default="", help="비우면 scancel 하지 않고 관측만")
    ap.add_argument("--simulate", action="store_true",
                    help="과거 로그를 재생해 언제 울렸을지 계산 (폴링·scancel 없음)")
    ap.add_argument("--stop-step", type=int, default=0, help="이 step 도달 시 정상 종료 (0=무제한)")
    #  아래 두 기본값은 추측이 아니라 73924/73925 로그 재생 스윕으로 고른 것이다(--simulate).
    #    임계 0.90 → step 287 오경보 / 0.85 → step 901 정확 / 0.70 → step 904 (늦음)
    #    창   20   → 붕괴前 최저 0.8734(마진 +0.023) / 50 → 0.8837(마진 +0.034)
    #  오경보는 잡을 죽여 런 전체를 잃고, 지연은 step 몇 개(≈5분/step)뿐이다 → 마진을 산다.
    ap.add_argument("--fmt-threshold", type=float, default=0.85,
                    help="FormatThink 창평균이 이 값 미만이면 붕괴 (73924 정상 구간 0.90~0.96)")
    ap.add_argument("--fmt-window", type=int, default=50, help="FormatThink 이동창 (step)")
    ap.add_argument("--fmt-min-samples", type=int, default=25,
                    help="창 안에 최소 이만큼 모여야 판정 — 재개 직후 오탐 방지")
    ap.add_argument("--kl-ratio", type=float, default=2.0)
    ap.add_argument("--kl-window", type=int, default=20)
    ap.add_argument("--kl-baseline", type=int, default=150,
                    help="현재 창 바로 앞 이만큼의 step 중앙값을 이동 기준선으로 쓴다")
    ap.add_argument("--kl-stable", type=float, default=1.3,
                    help="기준선 창의 후반/전반 중앙값 비가 이 값 이하일 때만 경보 무장(평탄 판정)")
    ap.add_argument("--kl-baseline-value", type=float, default=0.0,
                    help="기준선 직접 지정 (0=자동). 73924 의 step 700~849 평탄값 = 0.031")
    ap.add_argument("--poll", type=int, default=120, help="폴링 간격(초)")
    ap.add_argument("--timeout", type=int, default=0, help="이 초 지나면 감시 포기 (0=무제한)")
    a = ap.parse_args()
    (simulate if a.simulate else live)(a)


if __name__ == "__main__":
    main()
