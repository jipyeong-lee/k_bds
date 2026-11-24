#!/usr/bin/env python3
"""판(dont-get-lost)의 Design 탭과 「실험 결과」 탭을 만든다. 표준 라이브러리만 쓴다.

만드는 것: design/data.json · design/A_pipeline.svg(학습 흐름 그림) · design/results.md(평가 결과표) · design/calendar.md(달력).
design/ 은 생성물이다(gitignore). 손으로 고치지 말고 이 파일을 고친다.
원천: dashboard/data_src/days.json(달력) · dashboard/data_src/eval_finals.json · loco_eval.json(측정값) · dashboard/build_data.py 의 MODELS(모델 설명).
사용: python3 scripts/board_design.py && focus check
"""
from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "design"
FINALS = ROOT / "dashboard" / "data_src" / "eval_finals.json"
LOCO = ROOT / "dashboard" / "data_src" / "loco_eval.json"
DAYS = ROOT / "dashboard" / "data_src" / "days.json"   # [날짜, 그날 핵심 한 줄, [항목…]] 최신이 앞
RESULTS_HREF = "/p/kbds/results.md"   # data.json 의 link 는 그대로 쓰여서 프로젝트 접두어가 필요하다
PLOT_BASE = "/p/kbds/"                 # results.md 의 그림 주소도 같은 까닭으로 접두어를 붙인다

sys.path.insert(0, str(ROOT / "dashboard"))
from build_data import MODELS  # noqa: E402  (키, 표시 이름, 설명)

BOX_W, BOX_H = 160, 56
COLOUR = {"measured": "#0F766E", "neutral": "#334155", "open": "#DC2626"}
SIDES = {"measured": {"label": "평가함"}, "neutral": {"label": "학습 중 · 평가 전"}, "open": {"label": "정하지 않음"}}

# 번호: (상자 x, 상자 y, 제목, 부제, side, eval_finals 키 접두어 또는 None)
NODES = {
    1: (20, 120, "원본 Qwen3.5-9B", "학습 전 공개 모델", "measured", "base"),
    2: (215, 40, "① 콜드스타트 SFT", "sft_mixed_merged", "neutral", None),
    3: (215, 200, "풀이 줄이기 8차 75", "질 채점 짝 RPO · 새 시작 모델", "measured", None),
    4: (410, 40, "② deepvision 전문가", "GRPO 1,000 step", "neutral", None),
    5: (410, 120, "② mmk12 전문가", "GRPO 15k", "open", None),
    6: (410, 200, "② pmcvqa 전문가", "GRPO 20k", "open", None),
    7: (605, 120, "전문가 병합", "지금 기준 ta_w07", "measured", "ta_w07"),
    8: (800, 120, "③ 의료 특화 RL", "RaR 루브릭 보상", "open", None),
    9: (995, 120, "④ 평가", "홀드아웃 · 글로벌 벤치", "measured", None),
}
LOCO_NODE = {3: "8차 75단계"}  # 전량 평가가 없는 상자. loco_eval.json 의 이름으로 값을 붙인다
EDGES = [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7), (7, 8), (8, 9)]
DESC = {
    1: "학습하지 않은 공개 모델이다. 모든 비교의 바닥이다.",
    2: "`<think>/<answer>` 형식을 넣는다. 지난 Stage-2 의 시작 모델이었다.",
    3: "원본에 풀이 줄이기 8차 75 step LoRA 를 병합했다. Stage-2 다시 하기의 시작 모델이다. 엄격 형식을 지킨 답이 63~67% 라서 골랐다.",
    4: "deepvision 어려운 문항 2,010개로 1,000 step GRPO 를 한다. KISTI 에서 본학습한다.",
    5: "mmk12 15,204문항 GRPO. 8차 75 step 에서 시작할지 정하지 않았다.",
    6: "pmcvqa 19,583문항 GRPO. 8차 75 step 에서 시작할지 정하지 않았다.",
    7: "전문가 세 개를 가중치 0.7 로 병합한다. 지금 기준은 ta_w07 이다.",
    8: "medix-rl-data 51K, GRPO + RaR 루브릭 보상. 실행 장소와 보상 규칙을 정하지 않았다.",
    9: "홀드아웃 1,772문항과 글로벌 벤치 3,712문항으로 잰다. 결과는 「실험 결과」 탭에 있다.",
}


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}"


def scores(finals: dict, key: str) -> str:
    hold, gb = finals.get(f"{key}_hold", {}), finals.get(f"{key}_gb", {})
    return f"홀드아웃 **{pct(hold.get('accuracy'))}%** · 글로벌 벤치 **{pct(gb.get('accuracy'))}%**"


def node_svg(n: int) -> str:
    x, y, title, sub, side, _ = NODES[n]
    c = COLOUR[side]
    bx, by = x + BOX_W, y
    return (f'  <g class="node" data-c="{n}" data-x="{x}" data-y="{y}" data-w="{BOX_W}" data-h="{BOX_H}">\n'
            f'    <rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" fill="#fff" stroke="{c}" stroke-width="1.6"/>\n'
            f'    <text x="{x + 12}" y="{y + 24}" font-size="13" font-weight="600">{title}</text>\n'
            f'    <text x="{x + 12}" y="{y + 43}" font-size="11" fill="#64748B">{sub}</text>\n'
            f'  </g>\n'
            f'  <g class="badge" data-c="{n}">\n'
            f'    <circle cx="{bx}" cy="{by}" r="11" fill="{c}"/>\n'
            f'    <circle cx="{bx}" cy="{by}" r="18" fill="transparent"/>\n'
            f'    <text x="{bx}" y="{by + 4}" text-anchor="middle" font-size="11" fill="#fff">{n}</text>\n'
            f'  </g>\n')


def edge_svg(a: int, b: int) -> str:
    ax, ay = NODES[a][0] + BOX_W, NODES[a][1] + BOX_H // 2
    bx, by = NODES[b][0], NODES[b][1] + BOX_H // 2
    return f'  <path d="M{ax} {ay} L{bx} {by}" stroke="#94A3B8" stroke-width="1.4" fill="none"/>\n'


def pipeline_svg() -> str:
    w, h = 1190, 320
    legend = " · ".join(SIDES[s]["label"] for s in COLOUR)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="system-ui, sans-serif">\n'
            + "".join(edge_svg(a, b) for a, b in EDGES)
            + "".join(node_svg(n) for n in NODES)
            + f'  <g class="legend"><text x="20" y="300" font-size="11" fill="#64748B">테두리 색 · {legend}</text></g>\n'
            + "</svg>\n")


def loco_scores(loco: dict, name: str) -> str:
    acc = {k: next((r.get("acc") for r in rows if r["name"] == name), None) for k, rows in loco.items()}
    return f"풀이 줄이기 평가 수학 **{pct(acc.get('math'))}%** · 의료 **{pct(acc.get('med'))}%**"


def data_json(finals: dict, loco: dict) -> dict:
    comps = {}
    for n, (_, _, title, _, side, key) in NODES.items():
        desc = DESC[n] + (f" {scores(finals, key)}." if key else "")
        if n in LOCO_NODE:
            desc += f" {loco_scores(loco, LOCO_NODE[n])}."
        comps[str(n)] = {"label": title, "desc": desc, "side": side}
    comps["9"]["link"] = RESULTS_HREF
    return {
        "diagrams": {"A": {"file": "A_pipeline.svg", "title": "학습 흐름 · 원본에서 평가까지",
                           "ids": list(NODES),
                           "prose": "Stage-2 는 **풀이 줄이기 8차 75 step** 에서 다시 시작한다. "
                                    "SFT(2)는 지난 Stage-2 의 시작 모델이다.\n\n"
                                    "숫자는 `eval_finals.json` · `loco_eval.json` 에서 읽는다. 표 전체는 「실험 결과」 탭에 있다."}},
        "components": comps,
        "blocks": [{"lo": 1, "hi": 3, "name": "시작 모델"}, {"lo": 4, "hi": 7, "name": "Stage-2 전문가와 병합"},
                   {"lo": 8, "hi": 9, "name": "Stage-3 와 평가"}],
        "colour": COLOUR,
        "sides": SIDES,
    }


def results_md(finals: dict) -> str:
    rows = []
    for key, name, desc in MODELS:
        hold, gb = finals.get(f"{key}_hold"), finals.get(f"{key}_gb")
        if not hold and not gb:
            continue
        hold, gb = hold or {}, gb or {}
        src = hold.get("per_source", {})
        rows.append((hold.get("accuracy") or 0, [
            f"**{name}**" + (" (기준)" if key == "ta_w07" else ""),
            f"**{pct(hold.get('accuracy'))}**", f"**{pct(gb.get('accuracy'))}**",
            pct(src.get("deepvision")), pct(src.get("mmk12")), pct(src.get("pmcvqa")),
            pct(hold.get("format")), f"{hold.get('mean_chars', 0):,.0f}", desc]))
    head = ["모델", "홀드아웃", "글로벌 벤치", "deepvision", "mmk12", "pmcvqa", "형식 통과", "평균 글자 수", "설명"]
    table = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    table += ["| " + " | ".join(r) + " |" for _, r in sorted(rows, key=lambda t: -t[0])]
    n_hold = next((v["n"] for k, v in finals.items() if k.endswith("_hold")), "—")
    n_gb = next((v["n"] for k, v in finals.items() if k.endswith("_gb")), "—")
    return (f"# 실험 결과\n\n생성 {datetime.now():%Y-%m-%d %H:%M} · 생성물이다. 고칠 곳은 `scripts/board_design.py` 다.\n\n"
            f"정확도는 % 다. 홀드아웃은 {n_hold}문항, 글로벌 벤치는 {n_gb}문항이다. "
            "deepvision · mmk12 · pmcvqa 와 형식 통과 · 평균 글자 수는 홀드아웃 값이다. 홀드아웃 순서로 늘어놓는다.\n\n"
            + f"![전량 평가 정확도]({PLOT_BASE}finals.svg)\n\n"
            + "\n".join(table)
            + "\n\n원천: `dashboard/data_src/eval_finals.json`. 모델 설명: `dashboard/build_data.py` 의 `MODELS`.\n")


def vs_base(r: dict) -> str:
    v = r.get("vs_base")
    if not v:
        return "기준"
    sig = "유의함" if v["lo"] > 0 else "유의하게 낮음" if v["hi"] < 0 else "유의하지 않음"
    return f"{v['diff'] * 100:+.1f} [{v['lo'] * 100:+.1f}, {v['hi'] * 100:+.1f}] {sig}"


def loco_md(loco: dict) -> str:
    """풀이 줄이기 차수별 평가. 대시보드 「풀이 줄이기 평가」 표와 같은 값이다."""
    titles = {"math": "수학 그림 문제 (deepvision 150문항)", "med": "의료 (MedXpertQA-MM · MMMU 보건·의학 200문항)"}
    head = ["모델", "정확도", "원본 대비 (포인트, 95% 구간)", "평균 단어 수", "길이 상한 도달"]
    out = ["\n## 풀이 줄이기 차수별 평가\n",
           "같은 문항끼리 비교한 정확도 차이다. 구간은 문항 단위 부트스트랩이다. 의료는 답의 보기 글자만 비교해 다시 채점했다.\n",
           f"![풀이 길이와 정확도]({PLOT_BASE}loco_tradeoff.svg)\n",
           "점 하나가 평가 하나다. 오른쪽일수록 풀이가 길고, 위쪽일수록 정확하다.\n"]
    for key, rows in loco.items():
        out += [f"\n### {titles.get(key, key)}\n", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in rows:
            if r.get("acc") is None:
                out.append(f"| {r['name']} | " + ("하지 않았다" if r.get("skip") else "평가 전") + " | | | |")
                continue
            out.append(f"| {r['name']} | **{pct(r['acc'])}** | {vs_base(r)} | {r['words']:,.0f} | {pct(r.get('cap'))}% |")
    return "\n".join(out) + "\n\n원천: `dashboard/data_src/loco_eval.json`.\n"


# ---- 성능 그림. 표준 라이브러리만 쓰므로 SVG 를 직접 쓴다 ----
GRID, MUTED = "#E2E8F0", "#64748B"
# 실험 묶음마다 색과 모양을 함께 바꾼다(색만으로 가르지 않는다). 색은 Okabe-Ito.
FAMILY_STYLE = {"원본": ("#000000", "circle"), "1~5차": ("#E69F00", "square"), "6차": ("#0072B2", "circle"),
                "6차 변형": ("#56B4E9", "triangle"), "7차": ("#D55E00", "diamond"), "8차": ("#CC79A7", "square"),
                "SFT 모델": ("#009E73", "triangle")}
# 그림에 이름을 붙일 점: (글자, x 어긋남, y 어긋남). x 어긋남이 음수면 점 왼쪽에 쓴다.
LOCO_LABELS = {"원본 (Qwen3.5-9B, 학습 전)": ("원본", 8, -8), "5차": ("5차", 8, 4),
               "SFT 모델": ("SFT 모델", 8, 4), "6차 75단계": ("6차 75", -8, -6), "7차 75단계": ("7차 75", -8, 4)}
LOCO_TITLES = {"math": "수학 그림 문제 · deepvision 150문항", "med": "의료 · MedXpertQA-MM · MMMU 200문항"}


def family(name: str) -> str:
    if name.startswith("원본"):
        return "원본"
    if name.startswith("SFT"):
        return "SFT 모델"
    if name.startswith("6차 ") and any(v in name for v in ("형식", "RPO", "SFT")):
        return "6차 변형"
    return name.split()[0] if " " in name else "1~5차"


def marker(shape: str, x: float, y: float, c: str, r: float = 4.5) -> str:
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{c}"/>'
    if shape == "square":
        return f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" fill="{c}"/>'
    k = r * 1.3
    pts = [(x, y - k), (x + k, y), (x, y + k), (x - k, y)] if shape == "diamond" else [(x, y - k), (x + k, y + r), (x - k, y + r)]
    return f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="{c}"/>'


def text(x: float, y: float, s: str, size: int = 11, fill: str = MUTED, anchor: str = "start", extra: str = "") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{extra}>{html.escape(s)}</text>'


def grid(box: tuple, xs, ys, sx, sy, xfmt, yfmt) -> str:
    """눈금선과 눈금 글자. box 는 (x0, y0, 너비, 높이), sx · sy 는 값을 좌표로 바꾼다."""
    x0, y0, w, h = box
    out = [f'<line x1="{sx(v):.1f}" y1="{y0}" x2="{sx(v):.1f}" y2="{y0 + h}" stroke="{GRID}"/>'
           + text(sx(v), y0 + h + 16, xfmt(v), anchor="middle") for v in xs]
    out += [f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x0 + w}" y2="{sy(v):.1f}" stroke="{GRID}"/>'
            + text(x0 - 6, sy(v) + 4, yfmt(v), anchor="end") for v in ys]
    return "".join(out)


def legend(items: list, x: float, y: float, step: int = 84) -> str:
    return "".join(marker(s, x + i * step, y - 4, c) + text(x + i * step + 9, y, name)
                   for i, (name, c, s) in enumerate(items))


def svg(w: int, h: int, body: list) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="system-ui, sans-serif">\n<rect width="{w}" height="{h}" fill="#fff"/>\n'
            + "\n".join(body) + "\n</svg>\n")


def loco_svg(loco: dict) -> str:
    """풀이 줄이기: 평균 단어 수(x)와 정확도(y). 수학과 의료를 나란히 둔다. 두 판은 축 범위가 같다."""
    pw, ph, left, top, gap = 400, 280, 60, 40, 70
    x_hi, y_lo, y_hi = 4000, 0.15, 0.65
    w, h = left + 2 * pw + gap + 20, top + ph + 80
    body = []
    for i, key in enumerate(("math", "med")):
        rows = [r for r in loco.get(key, []) if r.get("acc") is not None]
        x0 = left + i * (pw + gap)
        sx = lambda v, x0=x0: x0 + v / x_hi * pw  # noqa: E731
        sy = lambda v: top + ph - (v - y_lo) / (y_hi - y_lo) * ph  # noqa: E731
        body.append(grid((x0, top, pw, ph), range(0, x_hi + 1, 1000), (0.2, 0.3, 0.4, 0.5, 0.6), sx, sy,
                         lambda v: f"{v:,}", lambda v: f"{v * 100:.0f}"))
        body.append(text(x0, top - 14, LOCO_TITLES.get(key, key), size=13, fill="#0F172A", extra=' font-weight="600"'))
        body.append(text(x0 + pw / 2, top + ph + 36, "평균 단어 수", anchor="middle"))
        base = next((r["acc"] for r in rows if family(r["name"]) == "원본"), None)
        if base is not None:
            body.append(f'<line x1="{x0}" y1="{sy(base):.1f}" x2="{x0 + pw}" y2="{sy(base):.1f}" '
                        f'stroke="#000" stroke-dasharray="4 3" stroke-width="1"/>')
        for r in rows:
            c, s = FAMILY_STYLE[family(r["name"])]
            body.append(marker(s, sx(r["words"]), sy(r["acc"]), c))
            if r["name"] in LOCO_LABELS:
                label, dx, dy = LOCO_LABELS[r["name"]]
                body.append(text(sx(r["words"]) + dx, sy(r["acc"]) + dy, label, fill="#0F172A",
                                 anchor="end" if dx < 0 else "start"))
    body.append(text(16, top + ph / 2, "정확도 (%)", anchor="middle",
                     extra=f' transform="rotate(-90 16 {top + ph / 2:.0f})"'))
    body.append(legend([(n, c, s) for n, (c, s) in FAMILY_STYLE.items()], left, h - 12)
                + text(left + 7 * 84, h - 12, "점선 = 원본 정확도"))
    return svg(w, h, body)


def finals_svg(finals: dict) -> str:
    """전량 평가 모델의 홀드아웃과 글로벌 벤치 정확도. 홀드아웃 순서로 늘어놓는다."""
    rows = [(name, finals.get(f"{k}_hold", {}).get("accuracy"), finals.get(f"{k}_gb", {}).get("accuracy"))
            for k, name, _ in MODELS if f"{k}_hold" in finals or f"{k}_gb" in finals]
    rows.sort(key=lambda r: -(r[1] or 0))
    left, top, pw, rh, lo, hi = 190, 20, 480, 30, 0.40, 0.80
    ph = len(rows) * rh
    w, h = left + pw + 30, top + ph + 70
    sx = lambda v: left + (v - lo) / (hi - lo) * pw  # noqa: E731
    body = [grid((left, top, pw, ph), (0.4, 0.5, 0.6, 0.7, 0.8), (), sx, None, lambda v: f"{v * 100:.0f}", None),
            text(left + pw / 2, top + ph + 36, "정확도 (%)", anchor="middle")]
    style = {"hold": ("#0072B2", "circle"), "gb": ("#E69F00", "square")}
    for i, (name, hold, gb) in enumerate(rows):
        y = top + i * rh + rh / 2
        body.append(text(left - 10, y + 4, name, size=12, fill="#0F172A", anchor="end"))
        if hold is not None and gb is not None:
            body.append(f'<line x1="{sx(hold):.1f}" y1="{y}" x2="{sx(gb):.1f}" y2="{y}" stroke="#CBD5E1" stroke-width="2"/>')
        for v, key in ((hold, "hold"), (gb, "gb")):
            if v is not None:
                body.append(marker(style[key][1], sx(v), y, style[key][0]))
    body.append(legend([("홀드아웃", *style["hold"]), ("글로벌 벤치", *style["gb"])], left, h - 12))
    return svg(w, h, body)


def calendar_md(days: list) -> str:
    """한 줄이 한 주(월~일)다. 칸 = 날짜 · 그날 핵심 한 줄 · 항목 수. 기록 없는 날은 날짜만."""
    from datetime import date, timedelta
    by_day = {d[0]: d for d in days}
    first, last = date.fromisoformat(min(by_day)), date.fromisoformat(max(by_day))

    def cell(d: date) -> str:
        if not first <= d <= last:
            return ""
        row = by_day.get(d.isoformat())
        if not row:
            return f"**{d:%m-%d}**"
        n = len(row[2])
        return f"**{d:%m-%d}** · {n}건 — " + row[1].replace("|", "/")

    monday = first - timedelta(days=first.weekday())
    weeks = [[cell(monday + timedelta(days=w * 7 + i)) for i in range(7)] for w in range((last - monday).days // 7 + 1)]
    head = ["월", "화", "수", "목", "금", "토", "일"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * 7] + ["| " + " | ".join(r) + " |" for r in weeks]
    return ("# 달력\n\n생성물이다. 고칠 곳은 `scripts/board_design.py` 다. 원천은 대시보드의 날짜별 기록(`dashboard/note.py`)이다.\n\n"
            "날마다 한 칸이다. 그날 한 일 한 줄과 기록한 항목 수를 보인다.\n\n" + "\n".join(lines) + "\n")


def main() -> int:
    if not FINALS.exists():
        print(f"원천이 없다: {FINALS}", file=sys.stderr)
        return 1
    finals = json.loads(FINALS.read_text(encoding="utf-8"))
    DESIGN.mkdir(exist_ok=True)
    (DESIGN / "A_pipeline.svg").write_text(pipeline_svg(), encoding="utf-8")
    (DESIGN / "finals.svg").write_text(finals_svg(finals), encoding="utf-8")
    loco = json.loads(LOCO.read_text(encoding="utf-8")) if LOCO.exists() else {}
    (DESIGN / "data.json").write_text(json.dumps(data_json(finals, loco), ensure_ascii=False, indent=1), encoding="utf-8")
    (DESIGN / "loco_tradeoff.svg").write_text(loco_svg(loco), encoding="utf-8")
    (DESIGN / "results.md").write_text(results_md(finals) + (loco_md(loco) if loco else ""), encoding="utf-8")
    days = json.loads(DAYS.read_text(encoding="utf-8")) if DAYS.exists() else []
    if days:
        (DESIGN / "calendar.md").write_text(calendar_md(days), encoding="utf-8")
    print(f"{DESIGN.relative_to(ROOT)}/ : A_pipeline.svg · finals.svg · loco_tradeoff.svg · data.json · results.md" + (" · calendar.md" if days else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
