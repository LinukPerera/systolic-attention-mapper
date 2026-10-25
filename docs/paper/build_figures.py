"""
build_figures.py — regenerates every animated SVG used by the root README.md.

    python docs/paper/build_figures.py            # all figures
    python docs/paper/build_figures.py fig02      # one figure (prefix match)

Data-driven figures read the project's own result files (experiments/exp*/results/), so every
plotted number traces to a logged run; schematic figures say so in their footer. The drawing
toolkit (type, colour tokens, looping build-up animation) lives in svgkit.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

import svgkit  # noqa: E402
from svgkit import (W, Cycle, Plate, arrow_head, circle, diamond, glyph, group, line, num, path, rect,  # noqa: E402
                    text, tspans, write)

OUT = HERE / "figures"
EXP = ROOT / "experiments"
DFTOK = {"WS": "ws", "OS": "os", "IS": "is"}           # dataflow -> colour token
DFKIND = {"WS": "ws", "OS": "os", "IS": "is"}          # dataflow -> glyph kind (square / disc / diamond)


def R(cy: Cycle, start: float, *parts: str, kind: str = "rise", dur: float = 0.7) -> str:
    """Bundle drawing parts into one group that joins the figure's rebuild at `start` seconds."""
    return group("".join(parts), anim=getattr(cy, kind)(start, dur))


def mn(s: str) -> str:
    return s.replace("-", "−")


def lerp(v: float, v0: float, v1: float, p0: float, p1: float) -> float:
    return p0 + (v - v0) * (p1 - p0) / (v1 - v0)


def flow_css(name: str, axis: str, dist: float, dur: float, period: float) -> str:
    """One-shot traversal inside a longer period: a mark slides `dist` px in `dur` s, then waits (hidden)."""
    f = 100 * dur / period
    t = "translateX" if axis == "x" else "translateY"
    return (f"@keyframes {name}{{0%{{transform:{t}(0);opacity:0}}2%{{opacity:1}}{num(f)}%{{transform:{t}({num(dist)}px);opacity:1}}"
            f"{num(f + .01)}%,100%{{transform:{t}({num(dist)}px);opacity:0}}}}")


def flash_css(name: str, period: float, rise: float = 0.10, fall: float = 0.55, lo: float = 0.0, hi: float = 1.0) -> str:
    return (f"@keyframes {name}{{0%{{opacity:{num(lo)}}}{num(100 * rise / period)}%{{opacity:{num(hi)}}}"
            f"{num(100 * fall / period)}%{{opacity:{num(lo)}}}100%{{opacity:{num(lo)}}}}}")


def triple(df_mode: str) -> list[str]:
    return df_mode.split("-")


# ======================================================================================
# Hero banner
# ======================================================================================
def fig00_hero() -> Plate:
    P, D = 6.0, 0.27          # period, seconds per systolic hop
    n, pitch, cell = 8, 30, 21
    x0, y0 = 668, 76
    v = pitch / D
    dist = 30 + n * pitch + 8
    p = Plate(436, "", "Title banner for the dissertation 'Learning-Based Dataflow Optimisation for Transformer "
              "Attention Kernels on Systolic Arrays', beside an 8 by 8 systolic array in which operands stream "
              "in from the left and top and each processing element lights as the two meet.",
              title="Learning-Based Dataflow Optimisation for Transformer Attention Kernels on Systolic Arrays",
              use_cycle=False, rules=True)
    p.extra_css = (flow_css("hx", "x", dist, dist / v, P) + flow_css("hy", "y", dist, dist / v, P)
                   + flash_css("hf", P) + ".fire{animation:hf %ss linear infinite}" % num(P))
    b = []
    b.append(text(48, 62, "MSc dissertation  ·  Electronics & Electrical Engineering  ·  Anglia Ruskin University", 11.5,
                  "sf", "mute", ls=1.3, caps=True))
    b.append(line(48, 78, 590, 78, stroke="ink", sw=0.75))
    b.append(text(48, 132, "Learning-Based Dataflow Optimisation", 33, "sf", "ink"))
    b.append(text(48, 174, "for Transformer Attention Kernels", 33, "sf", "ink"))
    b.append(text(48, 216, "on Systolic Arrays", 33, "sf", "acc", italic=True))
    b.append(tspans(48, 254, ["A hardware-in-the-loop study of per-stage dataflow mapping —",
                              "and what happens when the first formulation turns out to be wrong."],
                    15.5, "sf", "ink2", italic=True, lead=1.5))
    b.append(text(48, 318, "Linuk Perera", 15, "sf", "ink", weight=700))
    b.append(text(136, 318, "·  2026", 15, "sf", "mute"))
    stats = [("4 → 0", "cells where the learned mapper lost\nto the per-stage static baseline"),
             ("+9.5 %", "PE utilisation over the best single\ndataflow (seq. 256, 32×32 array)"),
             ("3,800 ×", "less simulator scratch data per\nevaluation after a one-line fix")]
    for i, (big, cap) in enumerate(stats):
        x = 48 + i * 292
        b.append(line(x, 372, x + 266, 372, stroke="hair", sw=0.75))
        b.append(text(x, 408, big, 28, "sf", "acc"))
        b.append(tspans(x + 118, 398, cap.split("\n"), 11.5, "sf", "mute", italic=True, lead=1.32))
    for i in range(n):
        for j in range(n):
            cx, cy_ = x0 + j * pitch, y0 + i * pitch
            b.append(rect(cx, cy_, cell, cell, stroke="faint", sw=0.9, rx=2))
            tij = 0.5 + (i + j) * D
            b.append(rect(cx, cy_, cell, cell, fill="acc", rx=2, op=0, cls="fire", anim=f"animation-delay:{num(tij - 0.1)}s"))
    for i in range(n):
        b.append(diamond(x0 - 30, y0 + i * pitch + cell / 2, 4.2, fill="is", op=0,
                         anim=f"animation:hx {num(P)}s linear infinite;animation-delay:{num(i * D)}s"))
    for j in range(n):
        b.append(rect(x0 + j * pitch + cell / 2 - 3.6, y0 - 33.6, 7.2, 7.2, fill="ws", op=0,
                      anim=f"animation:hy {num(P)}s linear infinite;animation-delay:{num(j * D)}s"))
    b.append(text(x0 + n * pitch - 9, y0 + n * pitch + 30, "weights ■ enter from the top, inputs ◆ from the left;", 11.5, "sf", "mute",
                  anchor="end", italic=True))
    b.append(text(x0 + n * pitch - 9, y0 + n * pitch + 46, "a PE fires on the clock cycle the two arrive together", 11.5, "sf", "mute",
                  anchor="end", italic=True))
    p.add(*b)
    return p


# ======================================================================================
# Fig. 1 — the problem: attention = three GEMMs, one array
# ======================================================================================
def fig01_problem() -> Plate:
    P = 7.2
    cy = Cycle(T=16, hold=3.4)
    p = Plate(478, "common/workload_gen.py · workload_to_gemm_shapes (schematic; array fill illustrative, not measured)",
              "Three columns, one per attention stage. Operand tokens flow left to right between stages. Under each stage "
              "an 8 by 8 array lights up where it can be used: all of it for the QKV projection, only half of the columns "
              "for the two per-head stages whose inner dimension is the head dimension.", cycle=cy,
              title="One attention block is three matrix multiplications, and one array cannot fill them all")
    pitch, cell = 21, 15
    cols = [(160, "Stage 1", "QKV projection", "(L × H) · (H × 3H)", 8, "100 %", "static weights"),
            (480, "Stage 2", "Scores  Q · K^{T}", "(L × D) · (D × L)", 4, "50 %", "both operands dynamic"),
            (800, "Stage 3", "Aggregate  scores · V", "(L × L) · (L × D)", 4, "50 %", "both operands dynamic")]
    starts = [0.4, 2.6, 4.8]
    css = [flash_css("f1", P, .12, .9, .22, .95), ".f1{animation:f1 %ss linear infinite}" % num(P)]
    for k in range(3):
        css.append(flow_css(f"t{k}", "x", 120, 0.9, P))
    p.extra_css = "".join(css)
    b = []
    for k, (cx, tag, name, shape, act, util, note) in enumerate(cols):
        ax = cx - 8 * pitch / 2
        ay = 150
        head = (text(cx, 66, tag, 11, "sf", "mute", anchor="middle", caps=True, ls=1.8)
                + text(cx, 92, name, 18, "sf", "ink", anchor="middle")
                + text(cx, 114, shape, 14, "sf", "ink2", anchor="middle", italic=True))
        b.append(R(cy, 0.2 + k * 0.5, head))
        cells = []
        for i in range(8):
            for j in range(8):
                x, y = ax + j * pitch, ay + i * pitch
                if j < act:
                    cells.append(rect(x, y, cell, cell, stroke="ink2", sw=0.9, rx=1.5))
                    cells.append(rect(x, y, cell, cell, fill="acc", rx=1.5, op=.22, cls="f1",
                                      anim=f"animation-delay:{num(starts[k] + 0.2 + (i + j) * 0.07)}s"))
                else:
                    cells.append(rect(x, y, cell, cell, stroke="faint", sw=0.8, rx=1.5, dash="2 2.5"))
        b.append(R(cy, 0.6 + k * 0.5, "".join(cells), kind="fade", dur=0.9))
        b.append(R(cy, 1.6 + k * 0.4,
                   line(ax, 336, ax + 8 * pitch - 6, 336, stroke="hair", sw=0.75),
                   text(cx, 372, util, 27, "sf", "acc", anchor="middle"),
                   text(cx, 394, note, 12.5, "sf", "mute", anchor="middle", italic=True)))
    gaps = [(250, 390, 0), (570, 710, 1)]
    for gx0, gx1, k in gaps:
        b.append(R(cy, 1.0 + k * 0.6, line(gx0, 222, gx1, 222, stroke="faint", sw=1), arrow_head(gx1 + 4, 222, 0, 6, "faint"), kind="fade"))
        for m in range(2):
            b.append(diamond(gx0 + 2, 222, 3.6, fill="is", op=0, anim=f"animation:t{k} {num(P)}s linear infinite;"
                             f"animation-delay:{num(starts[k + 1] - 1.0 + m * 0.18)}s"))
    b.append(R(cy, 1.4, text(640, 210, "softmax", 12, "sf", "mute", anchor="middle", italic=True)))
    b.append(R(cy, 1.4, line(40, 222, 70, 222, stroke="faint", sw=1), arrow_head(76, 222, 0, 6, "faint"),
               text(40, 210, "X", 13, "sf", "ink2", italic=True), line(890, 222, 920, 222, stroke="faint", sw=1),
               arrow_head(926, 222, 0, 6, "faint"), text(920, 210, "O", 13, "sf", "ink2", anchor="end", italic=True)))
    b.append(R(cy, 2.8, tspans(480, 436, ["With L = 256, H = 512 and n_{h} = 8 heads the head dimension is D = H / n_{h} = 64.",
                                          "On an array wider than D, stages 2 and 3 leave processing elements idle."],
                               13, "sf", "ink2", anchor="middle", italic=True, lead=1.45)))
    p.add(*b)
    return p


# ======================================================================================
# Fig. 2 — how data moves: the three stationary dataflows
# ======================================================================================
def fig02_dataflows() -> Plate:
    P, D = 6.4, 0.42
    pitch, cell = 40, 28
    v = pitch / D
    dist = 34 + 4 * pitch
    cy = Cycle(T=16, hold=3.4)
    p = Plate(462, "schematic · SCALE-Sim 3.0.0 implements exactly these three dataflows (no row-stationary mode)",
              "Three 4 by 4 processing-element grids. In each, one operand type stays resident in every PE "
              "while the other two stream through; a PE fires when the operands it needs arrive together.",
              cycle=cy, title="Three dataflows: which operand stays put, and which streams")
    p.extra_css = (flow_css("dx", "x", dist, dist / v, P) + flow_css("dy", "y", dist, dist / v, P) + flash_css("df", P, .10, .6, 0, .2)
                   + ".fire{animation:df %ss linear infinite}" % num(P)
                   + "@keyframes acc{0%{opacity:.18}8%{opacity:.18}70%{opacity:1}95%{opacity:1}100%{opacity:.18}}")
    panels = [
        ("Weight-stationary", "WS", "ws", [("is", "x"), ("os", "y")],
         ["Weights stay in the PEs; inputs flow through", "and partial sums gather as they move down.", "Suits stage 1, whose weights are static."]),
        ("Output-stationary", "OS", "os", [("is", "x"), ("ws", "y")],
         ["Each output stays put while inputs and weights", "both stream past it until the sum completes.", "No partial-sum traffic until the very end."]),
        ("Input-stationary", "IS", "is", [("ws", "x"), ("os", "y")],
         ["Inputs stay in the PEs; weights flow through", "and partial sums gather as they move down.", "Suits an input matrix that is reused heavily."]),
    ]
    b = []
    x_panels = [38, 350, 662]
    for k, (name, short, stat, movers, lines_) in enumerate(panels):
        px = x_panels[k]
        gx, gy = px + 52, 118
        parts = [text(px, 74, name, 17, "sf", "ink"), text(px + 258, 74, short, 12.5, "sf", "mute", anchor="end", italic=True),
                 line(px, 84, px + 258, 84, stroke="hair", sw=0.75)]
        for i in range(4):
            for j in range(4):
                cx, cy_ = gx + j * pitch, gy + i * pitch
                parts.append(rect(cx, cy_, cell, cell, stroke="faint", sw=0.9, rx=2))
                tij = 0.55 + (i + j) * D
                parts.append(rect(cx, cy_, cell, cell, fill="ink", rx=2, op=0, cls="fire", anim=f"animation-delay:{num(tij - .1)}s"))
                mx, my = cx + cell / 2, cy_ + cell / 2
                if stat == "os":
                    parts.append(circle(mx, my, 7, fill="os", anim=f"animation:acc {num(P)}s linear infinite;animation-delay:{num(tij - 0.55)}s"))
                else:
                    parts.append(glyph(DFKIND[short], mx, my, 5.6, stat))
        for i in range(4):
            tok = movers[0][0]
            parts.append(glyph(tok, gx - 30, gy + i * pitch + cell / 2, 3.6, tok, op=0,
                               anim=f"animation:dx {num(P)}s linear infinite;animation-delay:{num(i * D)}s"))
        for j in range(4):
            tok = movers[1][0]
            parts.append(glyph(tok, gx + j * pitch + cell / 2, gy - 30, 3.6, tok, op=0,
                               anim=f"animation:dy {num(P)}s linear infinite;animation-delay:{num(j * D)}s"))
        parts.append(tspans(px, 330, lines_, 13, "sf", "ink2", italic=True, lead=1.5))
        b.append(R(cy, 0.2 + k * 0.6, *parts, dur=0.9))
    for xx in (332, 644):
        b.append(line(xx, 70, xx, 392, stroke="hair", sw=0.75))
    leg = [text(38, 428, "Operands:", 12.5, "sf", "mute", italic=True), glyph("ws", 118, 424, 4.6, "ws"), text(130, 428, "weights  W", 13, "sf", "ink2"),
           glyph("is", 236, 424, 4.6, "is"), text(250, 428, "inputs  X", 13, "sf", "ink2"), glyph("os", 346, 424, 4.6, "os"),
           text(360, 428, "outputs / partial sums  Y", 13, "sf", "ink2"),
           text(922, 428, "large mark = held in the PE · small mark = streaming · shaded PE = firing", 12, "sf", "mute", anchor="end", italic=True)]
    b.append(R(cy, 2.2, *leg))
    p.add(*b)
    return p


# ======================================================================================
# search data shared by Fig. 3 and Fig. 4  (a real run: A2C, seed 2, seq 256 @ 32x32)
# ======================================================================================
def search_log() -> pd.DataFrame:
    raw = pd.read_csv(EXP / "exp3_main_sweep/results/raw/exp3_perstage.csv")
    c = raw[(raw.seq_len == 256) & (raw.array_rows == 32)]
    a = c[(c.optimizer == "a2c") & (c.seed == 2)].sort_values("iteration").copy()
    a["reward"] = a.notes.apply(lambda s: json.loads(s)["reward"])
    return a[["iteration", "dataflow_mode", "pe_utilization_pct", "reward"]].reset_index(drop=True)


def baselines_256_32() -> dict[str, float]:
    raw = pd.read_csv(EXP / "exp3_main_sweep/results/raw/exp3_perstage.csv")
    c = raw[(raw.seq_len == 256) & (raw.array_rows == 32)]
    return {m: float(c[c.optimizer == m].pe_utilization_pct.iloc[0]) for m in ("ws_only", "os_only", "is_only", "best_fixed_per_stage")}


# ======================================================================================
# Fig. 3 — the hardware-in-the-loop search (live)
# ======================================================================================
def fig03_loop() -> Plate:
    log = search_log()
    steps = len(log)
    STEP = 1.6
    T = steps * STEP
    p = Plate(430, "experiments/exp3_main_sweep/results/raw/exp3_perstage.csv · A2C, seed 2, seq 256 on a 32×32 array (real log)",
              "A search engine on the left proposes a mapping for the three attention stages; it crosses into a simulator "
              "on the right which returns measured utilisation; the number returns as a reward. The figure steps through the "
              "ten mappings of one real search run.", use_cycle=False,
              title="The hardware-in-the-loop search: every reward is a real simulation")
    css = [flow_css("pa", "x", 232, 0.55, STEP), flow_css("pr", "x", -232, 0.55, STEP), flash_css("sf", STEP, .08, .7, .35, 1)]
    for k in range(steps):
        a, b_ = 100 * k / steps, 100 * (k + 1) / steps
        css.append(f"@keyframes s{k}{{0%,{num(a)}%{{opacity:0}}{num(a + .01)}%,{num(b_)}%{{opacity:1}}{num(b_ + .01)}%,100%{{opacity:0}}}}")
    p.extra_css = "".join(css) + ".sim{animation:sf %ss linear infinite}" % num(STEP)
    b = []
    b.append(text(48, 70, "Search engine", 18, "sf", "ink"))
    b.append(line(48, 80, 330, 80, stroke="ink", sw=0.75))
    b.append(tspans(48, 104, ["A2C  (selected)  ·  PPO  ·  evolutionary", "Bayesian optimisation — all behind one interface"], 12.5, "sf", "mute", italic=True))
    b.append(text(630, 70, "Simulated hardware", 18, "sf", "ink"))
    b.append(line(630, 80, 912, 80, stroke="ink", sw=0.75))
    b.append(tspans(630, 104, ["SCALE-Sim 3.0.0 in a pinned Docker image", "one cycle-accurate run per attention stage"], 12.5, "sf", "mute", italic=True))
    b.append(line(480, 58, 480, 280, stroke="hair", sw=0.75, dash="3 4"))
    stage_lab = ["QKV", "Q·K^{T}", "scores·V"]
    for k in range(steps):
        row = log.iloc[k]
        df_ = triple(row.dataflow_mode)
        left, right = [], []
        left.append(text(48, 150, f"proposal {k + 1} of {steps}", 12.5, "sf", "ink2", italic=True))
        for s in range(3):
            x = 48 + s * 98
            left.append(text(x, 176, stage_lab[s], 12.5, "sf", "ink2"))
            left.append(glyph(DFKIND[df_[s]], x + 8, 202, 7.5, DFTOK[df_[s]]))
            left.append(text(x + 24, 207, df_[s], 15, "sf", DFTOK[df_[s]], weight=700))
        for s in range(3):
            x = 630 + s * 98
            right.append(text(x, 176, stage_lab[s], 12.5, "sf", "ink2"))
            for r_ in range(3):
                for c_ in range(3):
                    right.append(rect(x + c_ * 10, 188 + r_ * 10, 7, 7, fill=DFTOK[df_[s]], rx=1, op=.35, cls="sim",
                                      anim=f"animation-delay:{num(0.12 + (r_ + c_) * .06)}s"))
        right.append(text(630, 150, "measured on the simulated array", 12.5, "sf", "ink2", italic=True))
        right.append(text(630, 258, f"U = {row.pe_utilization_pct:.1f} %", 26, "sf", "ink"))
        b.append(group("".join(left), anim=f"animation:s{k} {num(T)}s linear infinite"))
        b.append(group("".join(right), anim=f"animation:s{k} {num(T)}s linear infinite"))
    b.append(line(340, 190, 616, 190, stroke="faint", sw=1))
    b.append(arrow_head(620, 190, 0, 6, "faint"))
    b.append(text(478, 180, "action", 12.5, "sf", "acc", anchor="middle", italic=True))
    b.append(line(620, 244, 344, 244, stroke="faint", sw=1))
    b.append(arrow_head(340, 244, 180, 6, "faint"))
    b.append(text(478, 266, "reward", 12.5, "sf", "warm", anchor="middle", italic=True))
    for m in range(3):
        b.append(rect(344 + m * 10, 186, 6, 8, fill="acc", rx=1, op=0, anim=f"animation:pa {num(STEP)}s linear infinite;animation-delay:{num(m * 0.05)}s"))
        b.append(circle(616 - m * 10, 244, 3.2, fill="warm", op=0, anim=f"animation:pr {num(STEP)}s linear infinite;animation-delay:{num(.75 + m * .05)}s"))
    b.append(line(48, 306, 912, 306, stroke="hair", sw=0.75))
    b.append(text(480, 346, "reward   r  =  U  −  λ · T / M", 19, "sf", "ink", anchor="middle", italic=True))
    b.append(text(480, 372, "U: measured PE utilisation   ·   T: SRAM↔DRAM traffic   ·   M: multiply-accumulates   ·   λ = 0.01", 12.5, "sf", "mute",
                  anchor="middle", italic=True))
    b.append(text(480, 398, "No surrogate model and no interpolation: a failed run is logged as failed, never dropped or replaced.", 13, "sf", "ink2",
                  anchor="middle", italic=True))
    p.add(*b)
    return p


# ======================================================================================
# Fig. 4 — one real search, step by step
# ======================================================================================
def fig04_trajectory() -> Plate:
    log = search_log()
    base = baselines_256_32()
    cy = Cycle(T=20, hold=3.6)
    p = Plate(508, "experiments/exp3_main_sweep/results/raw/exp3_perstage.csv (A2C, seed 2, seq 256 @ 32×32) and exp3_perstage_baseline_summary.csv",
              "A line of ten points showing the PE utilisation of each mapping a real search tried, labelled with its "
              "dataflow per stage. The run reaches input-stationary, weight-stationary, output-stationary at step four, "
              "above the exhaustive per-stage static baseline.", cycle=cy,
              title="Watching one search find its mapping")
    x0, x1, y0, y1 = 120, 860, 62, 372
    ylo, yhi = 66, 96
    X = lambda i: lerp(i, 0, 9, x0 + 30, x1 - 30)
    Y = lambda v: lerp(v, ylo, yhi, y1, y0)
    b = []
    ylabel = text(x0 - 56, (y0 + y1) / 2, "PE utilisation (%)", 12.5, "sf", "mute", anchor="middle", italic=True)
    ylabel = ylabel.replace('text-anchor="middle"', f'text-anchor="middle" transform="rotate(-90 {x0 - 56} {(y0 + y1) / 2})"')
    b.append(R(cy, 0.1, line(x0, y0, x0, y1, stroke="mute", sw=0.9), line(x0, y1, x1, y1, stroke="mute", sw=0.9),
               *[line(x0 - 4, Y(t), x0, Y(t), stroke="mute", sw=0.9) + text(x0 - 10, Y(t) + 4, str(t), 12, "sf", "ink2", anchor="end")
                 for t in range(70, 91, 10)],
               *[text(X(i), y1 + 20, str(i + 1), 12, "sf", "ink2", anchor="middle") for i in range(10)],
               text(x1 - 30, y1 + 40, "search step (one real simulation each)", 12.5, "sf", "mute", anchor="end", italic=True),
               ylabel, kind="fade"))
    refs = [("best_fixed_per_stage", "exhaustive per-stage static search (IS·WS·OS) — 87.45", "acc", "5 4"),
            ("os_only", "best single dataflow for the whole workload (OS) — 78.91", "mute", "2 4"),
            ("ws_only", "weight-stationary everywhere — 73.15", "faint", "2 4")]
    for k, (key, lab, tok, dash) in enumerate(refs):
        y = Y(base[key])
        b.append(R(cy, 0.4 + k * 0.3, line(x0, y, x1, y, stroke=tok, sw=1.1, dash=dash),
                   text(x1, y - 6, lab, 12, "sf", tok if tok != "faint" else "mute", anchor="end", italic=True, halo=True), kind="fade"))
    pts = [(X(i), Y(r.pe_utilization_pct)) for i, r in log.iterrows()]
    d = "M" + " L".join(f"{num(x)},{num(y)}" for x, y in pts)
    b.append(path(d, stroke="ink", sw=1.2, draw=True, anim=cy.draw(1.0, 6.0), op=.55))
    for i, r in log.iterrows():
        x, y = pts[i]
        df_ = triple(r.dataflow_mode)
        is_best = r.pe_utilization_pct == log.pe_utilization_pct.max()
        t_in = 1.0 + 6.0 * i / 9
        b.append(group(circle(x, y, 4.2 if not is_best else 5.4, fill="ink" if not is_best else "acc"), anim=cy.pop(t_in, 0.35), cls="pp"))
        prev = log.pe_utilization_pct[i - 1] if i > 0 else -1
        nxt = log.pe_utilization_pct[i + 1] if i < len(log) - 1 else -1
        is_peak = r.pe_utilization_pct >= max(prev, nxt)
        lab_y = y - 13 if is_peak else y + 24
        if not is_best:
            b.append(R(cy, t_in + 0.1, text(x, lab_y, "·".join(df_), 11.5, "sf", "ink2", anchor="middle", halo=True), kind="fade", dur=0.4))
    ib = int(log.pe_utilization_pct.idxmax())
    bx, by = pts[ib]
    b.append(R(cy, 7.8, circle(bx, by, 11, stroke="acc", sw=1.2),
               line(bx + 8, by - 8, bx + 46, by - 30, stroke="acc", sw=0.9),
               text(bx + 52, by - 34, f"step {ib + 1}:  IS·WS·OS  at  {log.pe_utilization_pct.max():.1f} %", 14.5, "sf", "acc", italic=True),
               text(bx + 52, by - 16, "above the exhaustive static search", 12.5, "sf", "ink2", italic=True), kind="fade", dur=0.8))
    b.append(R(cy, 8.6, tspans(480, 446, ["Three of the five seeds land on this same IS·WS·OS mapping —",
                                          "the per-stage choice that the exhaustive static search also makes."],
                               13.5, "sf", "ink2", anchor="middle", italic=True, lead=1.45)))
    p.add(*b)
    return p


# ======================================================================================
# small chart helpers
# ======================================================================================
def vtext(x: float, y: float, s: str, size: float = 12.5, fill: str = "mute") -> str:
    t = text(x, y, s, size, "sf", fill, anchor="middle", italic=True)
    return t.replace('text-anchor="middle"', f'text-anchor="middle" transform="rotate(-90 {num(x)} {num(y)})"')


def yaxis(x: float, y0: float, y1: float, ticks, Y, fmt=str, label: str = "") -> str:
    out = [line(x, y0, x, y1, stroke="mute", sw=0.9)]
    for t in ticks:
        out.append(line(x - 4, Y(t), x, Y(t), stroke="mute", sw=0.9))
        out.append(text(x - 9, Y(t) + 4, fmt(t), 12, "sf", "ink2", anchor="end"))
    if label:
        out.append(vtext(x - 52, (y0 + y1) / 2, label))
    return "".join(out)


def token_css(name: str, d: str, start: float, dur: float, period: float) -> str:
    s, e = 100 * start / period, 100 * (start + dur) / period
    return (f"@keyframes {name}{{0%,{num(s)}%{{offset-distance:0%;opacity:0}}{num(s + .3)}%{{opacity:1}}"
            f"{num(e)}%{{offset-distance:100%;opacity:1}}{num(e + .3)}%,100%{{offset-distance:100%;opacity:0}}}}")


# ======================================================================================
# Fig. 5 — the experimental programme as a validity chain
# ======================================================================================
def fig05_programme() -> Plate:
    cy = Cycle(T=18, hold=3.6)
    P = 9.0
    p = Plate(448, "dissertation Table 3.2 and §4.1.6 · each experiment is documented in experiments/expN_*/expN_stats.md",
              "A dependency graph of seven experiments. Experiment 0 selects the search strategy; Experiment 1 justifies the array "
              "sizes; Experiment 2 checks convergence; Experiment 3 is the main comparison; Experiments 4, 5 and 6 test scaling, "
              "ablation and transfer. A dashed arrow from Experiment 3 back to Experiment 0 marks the redesign that the main "
              "comparison forced.", cycle=cy, title="Seven experiments as a validity chain, with the arrow that sent the work back upstream")
    w_, h_ = 140, 84
    cols = [40, 226, 412, 598, 784]
    nodes = {0: (cols[0], 140, "Experiment 0", "Search-strategy\nbake-off", "which optimiser?"),
             1: (cols[0], 300, "Experiment 1", "Array-size\nsensitivity", "which array sizes?"),
             2: (cols[1], 140, "Experiment 2", "Search\nconvergence", "is the criterion sound?"),
             3: (cols[2], 220, "Experiment 3", "Main comparison\nvs. static mappers", "does it beat them?"),
             4: (cols[3], 140, "Experiment 4", "Scalability", "does it hold as L grows?"),
             5: (cols[3], 300, "Experiment 5", "Ablation", "which part does the work?"),
             6: (cols[4], 220, "Experiment 6", "Zero-shot\ntransfer", "does it generalise?")}
    edges = [(0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (3, 6)]
    def edge_path(a, c):
        xa, ya = nodes[a][0] + w_, nodes[a][1] + h_ / 2
        xc, yc = nodes[c][0], nodes[c][1] + h_ / 2
        mx = (xa + xc) / 2
        return f"M{num(xa)},{num(ya)} C{num(mx)},{num(ya)} {num(mx)},{num(yc)} {num(xc - 7)},{num(yc)}", (xc, yc)
    b = []
    css = []
    starts = {(0, 2): 0.4, (1, 3): 0.4, (2, 3): 1.6, (3, 4): 2.8, (3, 5): 2.8, (3, 6): 2.8}
    toks = []
    for k, (a, c) in enumerate(edges):
        d, (xe, ye) = edge_path(a, c)
        b.append(R(cy, 0.5 + k * 0.35, path(d, stroke="ink2", sw=1.1), arrow_head(xe, ye, 0, 6.5, "ink2"), kind="fade", dur=0.6))
        css.append(token_css(f"tk{k}", d, starts[(a, c)], 1.0, P))
        toks.append(circle(0, 0, 3.6, fill="acc", op=0, anim=f"offset-path:path('{d}');animation:tk{k} {num(P)}s linear infinite"))
    for n_, (x, y, tag, name, q) in nodes.items():
        stretch = n_ == 6
        parts = [rect(x, y, w_, h_, stroke="ink2" if not stretch else "mute", sw=1, dash="4 3" if stretch else ""),
                 text(x + 12, y + 19, tag, 10.5, "sf", "mute", caps=True, ls=1.2),
                 tspans(x + 12, y + 40, name.split("\n"), 14.5, "sf", "ink", lead=1.15),
                 text(x + 12, y + h_ - 10, q, 11.5, "sf", "mute", italic=True)]
        b.append(R(cy, 0.1 + n_ * 0.35, *parts))
    if True:
        b.append(R(cy, 4.2, text(cols[4] + w_ / 2, 210, "stretch goal", 11, "sf", "mute", anchor="middle", italic=True), kind="fade"))
    fx1, fy1 = cols[2] + w_ / 2, 220
    fx2, fy2 = cols[0] + w_ / 2, 140
    arc = f"M{num(fx1)},{num(fy1 - 2)} C{num(fx1)},{num(fy1 - 190)} {num(fx2)},{num(fy2 - 150)} {num(fx2)},{num(fy2 - 5)}"
    b.append(path(arc, stroke="warm", sw=1.6, dash="6 4", anim=cy.fade(4.4, 0.9)))
    b.append(R(cy, 4.6, arrow_head(fx2, fy2 - 4, 90, 7, "warm"), kind="fade", dur=0.5))
    css.append(token_css("tkb", arc, 4.6, 2.4, P))
    toks.append(circle(0, 0, 3.8, fill="warm", op=0, anim=f"offset-path:path('{arc}');animation:tkb {num(P)}s linear infinite"))
    b.append(R(cy, 4.8, text((fx1 + fx2) / 2, 102, "Experiment 3 finds the action space too coarse", 13.5, "sf", "warm", anchor="middle", italic=True),
               text((fx1 + fx2) / 2, 120, "→ redesign (3 → 9 numbers), re-run Experiments 0, 3, 4, 5", 12, "sf", "ink2", anchor="middle", italic=True)))
    b.append(R(cy, 5.4, text(480, 412, "Each experiment validates a precondition of the next; a problem found downstream sends the work back upstream "
                                       "instead of being reported unexamined.", 12.5, "sf", "ink2", anchor="middle", italic=True)))
    b.extend(toks)
    p.extra_css = "".join(css)
    p.add(*b)
    return p


# ======================================================================================
# Fig. 6 — Experiment 0: the optimiser ranking reverses
# ======================================================================================
def fig06_bakeoff() -> Plate:
    cy = Cycle(T=16, hold=3.4)
    o = csv_("exp0_optimizer_bakeoff/results/processed/exp0_summary.csv").set_index("optimizer")
    n = csv_("exp0_optimizer_bakeoff/results/processed/exp0_perstage_summary.csv").set_index("optimizer")
    p = Plate(492, "experiments/exp0_optimizer_bakeoff/results/processed/exp0_summary.csv and exp0_perstage_summary.csv (n = 5 seeds)",
              "A slope chart. Each of four search strategies is plotted by its distance from the brute-force optimum on the "
              "original three-number action space (left) and on the per-stage nine-number action space (right). Bayesian "
              "optimisation moves from best to worst; A2C and PPO move from the middle to joint best.", cycle=cy,
              title="The ranking of search strategies reverses when the action space is corrected")
    xl, xr = 330, 650
    Y = lambda v: lerp(v, 0, 4, 120, 380)
    b = []
    b.append(R(cy, 0.1, text(xl, 70, "original action space", 15, "sf", "ink", anchor="middle"),
               text(xl, 88, "3 numbers · one dataflow for the whole workload", 12, "sf", "mute", anchor="middle", italic=True),
               text(xr, 70, "per-stage action space", 15, "sf", "ink", anchor="middle"),
               text(xr, 88, "9 numbers · one dataflow per attention stage", 12, "sf", "mute", anchor="middle", italic=True),
               line(xl, 104, xl, 392, stroke="hair", sw=0.9), line(xr, 104, xr, 392, stroke="hair", sw=0.9), kind="fade"))
    b.append(R(cy, 0.2, yaxis(92, 110, 392, [0, 1, 2, 3, 4], Y, lambda t: str(t), "distance from brute-force optimum  (lower = better)"), kind="fade"))
    ao, ap = float(o.loc["a2c", "distance_from_brute_force_best"]), float(o.loc["ppo", "distance_from_brute_force_best"])
    an, pn = float(n.loc["a2c", "distance_from_brute_force_best"]), float(n.loc["ppo", "distance_from_brute_force_best"])
    series = [("Bayesian optimisation", float(o.loc["bayesian_opt", "distance_from_brute_force_best"]), float(n.loc["bayesian_opt", "distance_from_brute_force_best"]),
               "ink", 2.0, "5 / 5 seeds converged", "3 / 5 seeds converged", 0, 16),
              ("A2C · PPO", (ao + ap) / 2, (an + pn) / 2, "acc", 2.0, "", "5 / 5 seeds converged", 0, 0),
              ("Evolutionary", float(o.loc["evolutionary", "distance_from_brute_force_best"]), float(n.loc["evolutionary", "distance_from_brute_force_best"]),
               "mute", 1.4, "4 / 5 seeds converged", "5 / 5 seeds converged", 0, -20)]
    for k, (name, v0, v1, tok, sw, ann0, ann1, dyl, dyr) in enumerate(series):
        d = f"M{xl},{num(Y(v0))} L{xr},{num(Y(v1))}"
        b.append(path(d, stroke=tok, sw=sw, draw=True, anim=cy.draw(0.8 + k * 0.5, 1.2)))
        b.append(R(cy, 1.0 + k * 0.5, circle(xl, Y(v0), 5, fill=tok), circle(xr, Y(v1), 5, fill=tok), kind="fade", dur=0.4))
        lv0 = f"{ao:.2f} / {ap:.2f}" if name == "A2C · PPO" else f"{v0:.2f}"
        lv1 = f"{an:.2f} / {pn:.2f}" if name == "A2C · PPO" else f"{v1:.2f}"
        b.append(R(cy, 2.0 + k * 0.4,
                   text(xl - 14, Y(v0) + 4.5 + dyl, f"{name}   {lv0}", 13.5, "sf", tok if tok != "mute" else "ink2", anchor="end"),
                   text(xr + 14, Y(v1) + 4.5 + dyr, f"{lv1}   {name}", 13.5, "sf", tok if tok != "mute" else "ink2"), kind="fade"))
        if ann0:
            b.append(R(cy, 2.6 + k * 0.4, text(xl - 14, Y(v0) + 20 + dyl, ann0, 11.5, "sf", "mute", anchor="end", italic=True), kind="fade"))
        if ann1:
            b.append(R(cy, 2.6 + k * 0.4, text(xr + 14, Y(v1) + 20 + dyr, ann1, 11.5, "sf", "warm" if "3 /" in ann1 else "mute", italic=True), kind="fade"))
    b.append(R(cy, 3.8, tspans(480, 432, ["A2C and PPO are statistically indistinguishable, so A2C — the method the project originally specified — is retained",
                                          "for every later experiment. Bayesian optimisation's early lead did not survive the redesign."],
                               13, "sf", "ink2", anchor="middle", italic=True, lead=1.45)))
    p.add(*b)
    return p


def csv_(rel: str) -> pd.DataFrame:
    return pd.read_csv(EXP / rel)


# ======================================================================================
# Fig. 7 — Experiment 1: the utilisation knee tracks the head dimension
# ======================================================================================
def fig07_knee() -> Plate:
    cy = Cycle(T=18, hold=3.6)
    sizes = [8, 16, 32, 64, 128]
    d16 = csv_("exp1_array_sensitivity/results/processed/exp1_summary.csv")
    d32 = csv_("exp1_array_sensitivity/results/processed/exp1_representative_summary.csv")
    d64 = csv_("exp1_array_sensitivity/results/processed/exp1_head64_validation_summary.csv")
    def ser(df):
        s = df[df.method == "best_fixed_per_stage"].set_index("array_size").pe_utilization_pct
        return {a: float(s[a]) for a in sizes if a in s.index}
    S = {16: ser(d16), 32: ser(d32), 64: ser(d64)}
    p = Plate(528, "experiments/exp1_array_sensitivity/results/processed/ (exp1_summary, exp1_representative_summary, exp1_head64_validation_summary) · best per-stage static mapping",
              "PE utilisation against array size for three head dimensions. Utilisation falls slowly until the array size equals the "
              "head dimension, then falls steeply; doubling the head dimension moves that knee one array doubling to the right. "
              "Beneath the axis, small arrays show how much of an array of each size a head dimension of 32 can occupy.", cycle=cy,
              title="The utilisation knee sits where the array size reaches the head dimension")
    x0, x1, y0, y1 = 110, 880, 70, 330
    X = lambda i: lerp(i, 0, 4, x0 + 40, x1 - 100)
    Y = lambda v: lerp(v, 0, 100, y1, y0)
    b = []
    b.append(R(cy, 0.1, yaxis(x0, y0, y1, [0, 25, 50, 75, 100], Y, lambda t: str(t), "PE utilisation (%)"),
               line(x0, y1, x1 - 60, y1, stroke="mute", sw=0.9),
               *[line(X(i), y1, X(i), y1 + 4, stroke="mute", sw=0.9) + text(X(i), y1 + 20, f"{a}×{a}", 12, "sf", "ink2", anchor="middle") for i, a in enumerate(sizes)],
               text(x1 - 60, y1 + 38, "array size A", 12.5, "sf", "mute", anchor="end", italic=True), kind="fade"))
    cfg = {16: ("mute", "D = 16"), 32: ("acc", "D = 32"), 64: ("warm", "D = 64")}
    k = 0
    for D, (tok, lab) in cfg.items():
        pts = [(X(sizes.index(a)), Y(v)) for a, v in S[D].items()]
        d = "M" + " L".join(f"{num(x)},{num(y)}" for x, y in pts)
        b.append(path(d, stroke=tok, sw=1.8, draw=True, anim=cy.draw(0.8 + k * 1.2, 1.6)))
        b.append(R(cy, 1.0 + k * 1.2, *[circle(x, y, 3.6, fill=tok) for x, y in pts], kind="fade", dur=0.8))
        lx, ly = pts[-1]
        b.append(R(cy, 2.4 + k * 1.2, text(lx + 12, ly + 4, lab, 13.5, "sf", tok if tok != "mute" else "ink2", italic=True), kind="fade"))
        if D in (32, 64):   # the knee: where A reaches D
            kx = X(sizes.index(D))
            b.append(R(cy, 3.0 + k * 1.2, line(kx, y0 + 6, kx, y1, stroke=tok, sw=0.9, dash="3 4"),
                       text(kx - 6, y0 + 4, f"A = D = {D}", 12, "sf", tok, anchor="end", italic=True), kind="fade"))
        k += 1
    b.append(R(cy, 4.6, text(X(0) - 10, Y(S[64].get(16, 94)) - 70, "", 1), kind="fade"))
    # arrays: how much of an A x A array a head dimension of 32 can occupy
    b.append(R(cy, 5.0, text(x0, 388, "How much of an A × A array a stage capped at D = 32 can occupy:", 13, "sf", "ink2", italic=True), kind="fade"))
    for i, a in enumerate(sizes):
        frac = min(32, a) / a
        sz = 44
        cx = X(i) - sz / 2
        b.append(R(cy, 5.4 + i * 0.5, rect(cx, 402, sz, sz, stroke="ink2", sw=0.9),
                   group(rect(cx, 402, sz * frac, sz, fill="acc", op=.55, cls="gx"), anim=cy.grow_x(5.7 + i * 0.5, 0.6)),
                   text(cx + sz / 2, 468, f"{frac * 100:.0f} %", 14, "sf", "acc" if frac == 1 else "warm", anchor="middle"), kind="fade", dur=0.4))
    b.append(R(cy, 8.6, text((X(3) + X(4)) / 2, 494, "once A exceeds D, the extra PEs sit idle", 13, "sf", "warm", anchor="middle", italic=True)))
    b.append(R(cy, 4.8, text(X(0) + 12, Y(60), "D = 64 was not run at 8×8:", 11.5, "sf", "warm", italic=True),
               text(X(0) + 12, Y(60) + 15, "out of memory, logged as failed", 11.5, "sf", "warm", italic=True), kind="fade"))
    p.add(*b)
    return p


# ======================================================================================
# Fig. 8 — Experiment 2: a convergence criterion that fires too early
# ======================================================================================
def fig08_convergence() -> Plate:
    cy = Cycle(T=18, hold=3.6)
    tr = csv_("exp2_convergence/results/processed/exp2_trajectories.csv")
    piv = tr.pivot(index="iteration", columns="seed", values="best_reward_so_far")
    mean = piv.mean(axis=1)
    final = float(mean.iloc[-1])
    start_v = float(mean.iloc[0])
    flat = int(next(i for i, v in mean.items() if v >= start_v + 0.95 * (final - start_v)))
    p = Plate(492, "experiments/exp2_convergence/results/processed/exp2_trajectories.csv · five seeds, 40 real iterations each",
              "Running-best reward against search iteration for five seeds, with their mean. A stated convergence criterion declares "
              "all five seeds converged at iteration 4, yet the mean keeps improving until about iteration 25.", cycle=cy,
              title="A stated convergence criterion fires at iteration 4; the curve keeps rising to about iteration 25")
    x0, x1, y0, y1 = 110, 880, 70, 372
    lo, hi = 84.0, 86.4
    X = lambda i: lerp(i, 0, 39, x0 + 10, x1 - 20)
    Y = lambda v: lerp(v, lo, hi, y1, y0)
    b = []
    b.append(R(cy, 0.1, yaxis(x0, y0, y1, [84, 85, 86], Y, lambda t: f"{t:.0f}", "best reward so far"), line(x0, y1, x1, y1, stroke="mute", sw=0.9),
               *[line(X(i), y1, X(i), y1 + 4, stroke="mute", sw=0.9) + text(X(i), y1 + 20, str(i), 12, "sf", "ink2", anchor="middle") for i in (0, 10, 20, 30, 39)],
               text(x1 - 20, y1 + 40, "search iteration", 12.5, "sf", "mute", anchor="end", italic=True), kind="fade"))
    for s in piv.columns:
        d = "M" + " L".join(f"{num(X(i))},{num(Y(float(v)))}" for i, v in piv[s].items())
        b.append(path(d, stroke="faint", sw=1, draw=True, anim=cy.draw(0.6, 2.0), op=.9))
    d = "M" + " L".join(f"{num(X(i))},{num(Y(float(v)))}" for i, v in mean.items())
    b.append(path(d, stroke="ink", sw=2.2, draw=True, anim=cy.draw(0.8, 3.4)))
    kx = X(4)
    b.append(R(cy, 4.4, line(kx, y0 + 4, kx, y1, stroke="warm", sw=1.2, dash="5 4"),
               text(kx + 10, y0 + 18, "criterion fires: 5 of 5 seeds “converged”", 14, "sf", "warm", italic=True),
               text(kx + 10, y0 + 37, "at iteration 4 — and a 10× stricter threshold agrees", 12.5, "sf", "ink2", italic=True), kind="fade"))
    fx = X(flat)
    b.append(R(cy, 5.4, line(fx, Y(float(mean.iloc[flat])) + 4, fx, y1, stroke="acc", sw=1.2, dash="5 4"),
               text(fx - 10, y1 - 30, f"the mean has made 95 % of its gain only", 13.5, "sf", "acc", anchor="end", italic=True),
               text(fx - 10, y1 - 12, f"by iteration {flat}", 13.5, "sf", "acc", anchor="end", italic=True),
               kind="fade"))
    b.append(R(cy, 6.4, tspans(480, 430, ["The trailing-window variance test is satisfied by a low-variance run of early, seed-determined trials — a property of the",
                                          "criterion, not of the optimiser. Later experiments therefore use fixed, pre-declared budgets instead."],
                               13, "sf", "ink2", anchor="middle", italic=True, lead=1.45)))
    p.add(*b)
    return p


# ======================================================================================
# Fig. 9 — the redesign: one dataflow for everything vs. one per stage
# ======================================================================================
def best_action_256_32() -> tuple[dict, dict]:
    raw = pd.read_csv(EXP / "exp3_main_sweep/results/raw/exp3_perstage.csv")
    c = raw[(raw.seq_len == 256) & (raw.array_rows == 32)]
    a = c[(c.optimizer == "a2c") & (c.seed == 2)].copy()
    a["reward"] = a.notes.apply(lambda s: json.loads(s)["reward"])
    row = a.loc[a.reward.idxmax()]
    stages = json.loads(row.notes)["stages"]
    bfps = json.loads(c[c.optimizer == "best_fixed_per_stage"].iloc[0].notes)["per_stage_dataflow"]
    return stages, bfps


def fig09_redesign() -> Plate:
    cy = Cycle(T=16, hold=3.6)
    P = 6.4
    stages, bfps = best_action_256_32()
    base = baselines_256_32()
    log = search_log()
    names = [("qkv_projection", "QKV projection"), ("qk_transpose", "Scores  Q·K^{T}"), ("score_times_v", "Aggregate  scores·V")]
    bx = [140, 392, 644]
    bw, bh = 196, 74
    dist = 760
    p = Plate(528, "experiments/exp3_main_sweep/results/raw/exp3_perstage.csv · best action of A2C seed 2, seq 256 @ 32×32; os_only baseline",
              "Two pipelines of the three attention stages. In the original action space (top) one dataflow, here output-stationary, "
              "is applied to all three stages and they run in a single simulator call. In the redesigned per-stage action space "
              "(bottom) each stage has its own dataflow and SRAM split, each simulated separately; a token flows through both.",
              cycle=cy, title="From one dataflow for the whole workload to one dataflow per stage")
    css = [flow_css("tok", "x", dist, 4.4, P), flash_css("bx", P, .15, .9, 0, .16), ".bx{animation:bx %ss linear infinite}" % num(P)]
    p.extra_css = "".join(css)
    b = []

    def panel(y: float, tag: str, title: str, sub: str, per_stage: bool, k0: float) -> list[str]:
        out = [text(40, y, tag, 15, "sf", "ink", weight=700), text(66, y, title, 15, "sf", "ink"),
               text(66, y + 19, sub, 12.5, "sf", "mute", italic=True)]
        yb = y + 62
        out.append(text(40, yb + bh / 2 + 5, "X", 14, "sf", "ink2", italic=True))
        out.append(line(54, yb + bh / 2, 134, yb + bh / 2, stroke="faint", sw=1))
        out.append(line(840, yb + bh / 2, 900, yb + bh / 2, stroke="faint", sw=1))
        out.append(arrow_head(906, yb + bh / 2, 0, 6, "faint"))
        out.append(text(916, yb + bh / 2 + 5, "O", 14, "sf", "ink2", italic=True))
        for i, (key, label) in enumerate(names):
            df_ = bfps[key] if per_stage else "OS"
            tok = DFTOK[df_]
            x = bx[i]
            arrival = (x + bw / 2 - 54) / dist * 4.4
            out.append(rect(x, yb, bw, bh, stroke="ink2", sw=1))
            out.append(rect(x, yb, bw, bh, fill=tok, op=0, cls="bx", anim=f"animation-delay:{num(arrival - .15)}s"))
            out.append(text(x + 12, yb + 22, label, 14, "sf", "ink"))
            out.append(glyph(DFKIND[df_], x + 20, yb + 50, 6.5, tok))
            out.append(text(x + 36, yb + 55, df_, 16, "sf", tok, weight=700))
            if per_stage:
                st = stages[key]
                fr = [st["ifmap_frac"], st["filter_frac"], st["ofmap_frac"]]
                bx0, bw0 = x + 78, 104
                cx_ = bx0
                out.append(rect(bx0, yb + 45, bw0, 8, stroke="faint", sw=0.8))
                for f_, tk, opv in zip(fr, ("acc", "ink", "faint"), (.9, .55, .8)):
                    if f_ > 0.004:
                        out.append(rect(cx_, yb + 45, bw0 * f_, 8, fill=tk, op=opv))
                        cx_ += bw0 * f_
                out.append(text(bx0, yb + 66, "SRAM split", 10.5, "sf", "mute", italic=True))
            if i < 2:
                out.append(line(x + bw + 4, yb + bh / 2, bx[i + 1] - 8, yb + bh / 2, stroke="faint", sw=1))
                out.append(arrow_head(bx[i + 1] - 4, yb + bh / 2, 0, 6, "faint"))
        out.append(diamond(54, yb + bh / 2, 4.6, fill="is", op=0, anim=f"animation:tok {num(P)}s linear infinite"))
        return out

    # (a) original
    pa = panel(70, "(a)", "Original action space — 3 numbers", "one (dataflow, SRAM split) applied to the whole workload; one simulator call", False, 0.2)
    pa.append(line(bx[0], 120, bx[2] + bw, 120, stroke="warm", sw=1.2))
    pa.append(line(bx[0], 120, bx[0], 130, stroke="warm", sw=1.2)); pa.append(line(bx[2] + bw, 120, bx[2] + bw, 130, stroke="warm", sw=1.2))
    pa.append(text((bx[0] + bx[2] + bw) / 2, 113, "one shared choice for all three stages", 12.5, "sf", "warm", anchor="middle", italic=True))
    pa.append(text(900, 252, f"best reachable: OS everywhere — {base['os_only']:.1f} %", 13.5, "sf", "warm", anchor="end", italic=True))
    b.append(R(cy, 0.2, *pa, dur=0.9))
    b.append(line(40, 276, 920, 276, stroke="hair", sw=0.75))
    # (b) redesigned
    pb = panel(300, "(b)", "Per-stage action space — 9 numbers", "each stage gets its own (dataflow, SRAM split); three simulator calls, then combined", True, 1.2)
    pb.append(text(900, 470, f"found by search: {'·'.join(bfps[k] for k, _ in names)} — {log.pe_utilization_pct.max():.1f} %", 13.5, "sf", "acc", anchor="end", italic=True))
    b.append(R(cy, 2.0, *pb, dur=0.9))
    b.append(R(cy, 3.6, text(40, 490, "cost: three simulator calls per candidate instead of one", 12.5, "sf", "ink2", italic=True),
               text(40, 470, "SRAM split bar:", 11.5, "sf", "mute", italic=True), rect(124, 462, 9, 9, fill="acc", op=.9), text(137, 470, "ifmap", 11.5, "sf", "ink2"),
               rect(176, 462, 9, 9, fill="ink", op=.55), text(189, 470, "filter", 11.5, "sf", "ink2"), rect(229, 462, 9, 9, fill="faint", op=.8), text(242, 470, "ofmap", 11.5, "sf", "ink2"), kind="fade"))
    p.add(*b)
    return p


# ======================================================================================
# Fig. 10 — Experiment 3: the gap to the per-stage baseline closes
# ======================================================================================
def fig10_gap() -> Plate:
    cy = Cycle(T=18, hold=3.8)
    old = csv_("exp3_main_sweep/results/processed/exp3_optimizer_vs_baseline.csv")
    new = csv_("exp3_main_sweep/results/processed/exp3_perstage_optimizer_vs_baseline.csv")
    so = csv_("exp3_main_sweep/results/processed/exp3_significance.csv")
    sn = csv_("exp3_main_sweep/results/processed/exp3_perstage_significance.csv")
    cells = list(new.cell)
    rows_y = [150, 218, 286, 354]
    p = Plate(486, "experiments/exp3_main_sweep/results/processed/ (exp3_ and exp3_perstage_ optimizer_vs_baseline, significance) · n = 5 seeds · bars = 95 % CI",
              "Dumbbell charts for four workload and array cells. Left: the learned mapper's utilisation relative to the best per-stage "
              "static mapping; the original action space lost significantly in every cell and the per-stage action space is "
              "statistically level with it. Right: relative to the best single dataflow, the margin grows.", cycle=cy,
              title="Per-stage action space closes the gap to the per-stage static baseline")
    panels = [("(a)", "relative to the best per-stage static mapping", 168, 396, -10.5, 3.0, "best_overall", lambda c: so[(so.cell == c) & so.comparison.str.contains("best_overall")], lambda c: sn[(sn.cell == c) & sn.comparison.str.contains("best_overall")], "parity"),
              ("(b)", "relative to the best single dataflow", 570, 790, -1.0, 11.0, "best_single_dataflow", lambda c: so[(so.cell == c) & so.comparison.str.contains("best_single")], lambda c: sn[(sn.cell == c) & sn.comparison.str.contains("best_single")], "parity")]
    b = []
    for ci, c in enumerate(cells):
        lab = c.replace(" @ ", " · ").replace("seq", "sequence ")
        b.append(R(cy, 0.2 + ci * 0.15, text(40, rows_y[ci] + 5, lab, 13.5, "sf", "ink2"), kind="fade", dur=0.5))
    for pi, (tag, title, xa, xb, lo, hi, key, fo, fn, _) in enumerate(panels):
        X = lambda v: lerp(v, lo, hi, xa, xb)
        head = [text(xa - 30, 78, tag, 14, "sf", "ink", weight=700), text(xa, 78, title, 14.5, "sf", "ink"),
                line(xa - 8, 90, xb + 8, 90, stroke="ink", sw=0.6)]
        b.append(R(cy, 0.1, *head, kind="fade"))
        zero = X(0)
        ticks = [t for t in range(int(lo) if lo == int(lo) else int(lo) - 1, int(hi) + 1) if lo <= t <= hi and t % (2 if pi == 0 else 2) == 0]
        ax = [line(zero, 120, zero, 392, stroke="ink2", sw=0.9, dash="3 3"), text(zero, 112, "parity", 11.5, "sf", "mute", anchor="middle", italic=True)]
        ax.append(line(xa - 8, 392, xb + 8, 392, stroke="mute", sw=0.9))
        for t in ticks:
            ax.append(line(X(t), 392, X(t), 396, stroke="mute", sw=0.9))
            ax.append(text(X(t), 412, mn(f"{t:+d}") if t else "0", 12, "sf", "ink2", anchor="middle"))
        ax.append(text(xb + 8, 432, "Δ utilisation, %", 12, "sf", "mute", anchor="end", italic=True))
        b.append(R(cy, 0.4, *ax, kind="fade"))
        for ci, c in enumerate(cells):
            y = rows_y[ci]
            ro, rn = old[old.cell == c].iloc[0], new[new.cell == c].iloc[0]
            vo, vn = float(ro[f"{key}_pct_improvement_mean"]), float(rn[f"{key}_pct_improvement_mean"])
            co, cn = float(ro[f"{key}_pct_improvement_ci"]), float(rn[f"{key}_pct_improvement_ci"])
            po = float(fo(c).t_pvalue.iloc[0]); pn = float(fn(c).t_pvalue.iloc[0])
            b.append(R(cy, 1.0 + ci * 0.3, line(X(vo - co), y, X(vo + co), y, stroke="warm", sw=1.2), circle(X(vo), y, 5.4, fill="bg", stroke="warm", sw=1.8), kind="fade", dur=0.6))
            b.append(group(line(X(vn - cn), y + 11, X(vn + cn), y + 11, stroke="faint", sw=1.2) + circle(X(vn), y + 11, 5.6, fill="acc") +
                           line(X(vo), y + 0.5, X(vn), y + 11 - 0.5, stroke="acc", sw=0.7, op=0), anim=cy.slide(4.2 + ci * 0.4, X(vo) - X(vn), -11, 1.4)))
            fmt = lambda pv: ("p < 10^{−4}" if pv < 1e-4 else f"p = {pv:.3f}" if pv < 0.1 else f"p = {pv:.2f}")
            sig_o, sig_n = po < 0.05, pn < 0.05
            b.append(R(cy, 6.4 + ci * 0.25, text(xb + 24, y + 2, fmt(po), 11.5, "sf", "warm" if sig_o else "mute", italic=True),
                       text(xb + 24, y + 17, fmt(pn) + ("" if sig_n else "  n.s."), 11.5, "sf", "acc" if sig_n else "mute", italic=True), kind="fade"))
    b.append(R(cy, 0.9, circle(40 + 6, 452, 5, fill="bg", stroke="warm", sw=1.8), text(58, 456, "original action space (3 numbers)", 12.5, "sf", "ink2"),
               circle(300, 452, 5.4, fill="acc"), text(312, 456, "per-stage action space (9 numbers)", 12.5, "sf", "ink2"),
               text(924, 456, "upper p-value: original · lower: per-stage", 11.5, "sf", "mute", anchor="end", italic=True)))
    p.add(*b)
    return p


# ======================================================================================
# Fig. 11 — Experiment 4: the sequence-length reversal disappears
# ======================================================================================
def fig11_scale() -> Plate:
    cy = Cycle(T=18, hold=3.8)
    old = csv_("exp4_scalability/results/processed/exp4_scalability_summary.csv")
    new = csv_("exp4_scalability/results/processed/exp4_perstage_scalability_summary.csv")
    o = old[old.array_size == 32].sort_values("seq_len")
    n = new[new.array_size == 32].sort_values("seq_len")
    p = Plate(500, "experiments/exp4_scalability/results/processed/ (exp4_ and exp4_perstage_ scalability_summary) · 32×32 array, n = 5 seeds · bars = 95 % CI",
              "Left: improvement over the best single dataflow against sequence length. The original action space turns negative at "
              "sequence 512; the per-stage action space stays positive. Right: search wall-clock cost, which grows with sequence length "
              "and is higher for the per-stage action space because it makes three simulator calls per candidate.", cycle=cy,
              title="The reversal at longer sequences disappears; cost still grows")
    xs = {128: 150, 256: 300, 512: 450}
    Y = lambda v: lerp(v, -4, 14, 360, 100)
    b = []
    b.append(text(40, 78, "(a)", 14, "sf", "ink", weight=700)); b.append(text(66, 78, "improvement over the best single dataflow", 14.5, "sf", "ink"))
    b.append(line(34, 90, 540, 90, stroke="ink", sw=0.6))
    b.append(R(cy, 0.1, yaxis(100, 100, 360, [-4, 0, 4, 8, 12], Y, lambda t: mn(f"{t:+d}") if t else "0", "Δ utilisation, %"),
               line(100, Y(0), 520, Y(0), stroke="ink2", sw=0.9, dash="3 3"), line(100, 360, 520, 360, stroke="mute", sw=0.9),
               *[line(x, 360, x, 364, stroke="mute", sw=0.9) + text(x, 380, f"L = {L}", 12.5, "sf", "ink2", anchor="middle") for L, x in xs.items()],
               text(520, Y(0) - 7, "parity", 11.5, "sf", "mute", anchor="end", italic=True), kind="fade"))
    for tok, df_, lab, dx0, st in (("warm", o, "original action space", -4, 0.8), ("acc", n, "per-stage action space", 4, 2.8)):
        pts = [(xs[int(r.seq_len)] + dx0, Y(float(r.pct_improvement_mean)), float(r.pct_improvement_ci)) for _, r in df_.iterrows()]
        d = "M" + " L".join(f"{num(x)},{num(y)}" for x, y, _ in pts)
        b.append(path(d, stroke=tok, sw=1.8, draw=True, anim=cy.draw(st, 1.4)))
        wh = "".join(line(x, Y(float(v) - c_), x, Y(float(v) + c_), stroke=tok, sw=1, op=.7)
                     for (x, _, c_), v in zip(pts, df_.pct_improvement_mean))
        b.append(R(cy, st + .6, wh, *[circle(x, y, 4.4, fill=tok if tok == "acc" else "bg", stroke=tok, sw=1.6) for x, y, _ in pts], kind="fade", dur=0.8))
        ex, ey, _ = pts[-1]
        b.append(R(cy, st + 1.4, text(ex + 14, ey + 4, mn(f"{float(df_.pct_improvement_mean.iloc[-1]):+.1f} %"), 14, "sf", tok), kind="fade"))
    b.append(R(cy, 2.6, text(112, 330, "original action space", 12.5, "sf", "warm", italic=True), kind="fade"))
    b.append(R(cy, 4.6, text(112, 160, "per-stage action space", 12.5, "sf", "acc", italic=True), kind="fade"))
    # (b) cost bars
    b.append(text(600, 78, "(b)", 14, "sf", "ink", weight=700)); b.append(text(626, 78, "search cost, seconds of wall-clock", 14.5, "sf", "ink"))
    b.append(line(594, 90, 926, 90, stroke="ink", sw=0.6))
    Yc = lambda v: lerp(v, 0, 1200, 360, 100)
    b.append(R(cy, 0.2, yaxis(620, 100, 360, [0, 400, 800, 1200], Yc, lambda t: str(t)), line(620, 360, 910, 360, stroke="mute", sw=0.9), kind="fade"))
    for gi, L in enumerate((128, 256, 512)):
        gx = 665 + gi * 100
        co = float(o[o.seq_len == L].cost_mean.iloc[0]); cn = float(n[n.seq_len == L].cost_mean.iloc[0])
        b.append(R(cy, 0.4, text(gx + 22, 380, f"L = {L}", 12.5, "sf", "ink2", anchor="middle"), kind="fade"))
        b.append(rect(gx, Yc(co), 20, 360 - Yc(co), fill="none", stroke="warm", sw=1.2, cls="gy", anim=cy.grow_y(1.0 + gi * 0.4, 0.9)))
        b.append(rect(gx + 24, Yc(cn), 20, 360 - Yc(cn), fill="acc", cls="gy", anim=cy.grow_y(1.2 + gi * 0.4, 0.9)))
        b.append(R(cy, 2.2 + gi * 0.3, text(gx + 10, Yc(co) - 5, f"{co:.0f}", 11, "sf", "warm", anchor="middle"), text(gx + 34, Yc(cn) - 5, f"{cn:.0f}", 11, "sf", "acc", anchor="middle"), kind="fade"))
    b.append(R(cy, 3.4, tspans(640, 408, ["≈ 3 × the simulator calls per candidate:", "quality improved, cost did not."], 12.5, "sf", "ink2", italic=True, lead=1.4), kind="fade"))
    b.append(R(cy, 5.4, text(40, 460, "At L = 512 the original run lost to the best static dataflow (−1.6 %); five of five seeds now converge, with the same six-iteration budget.", 13, "sf", "ink2", italic=True)))
    p.add(*b)
    return p


# ======================================================================================
# Fig. 12 — Experiment 5: dataflow-only search now captures the gain
# ======================================================================================
def fig12_ablation() -> Plate:
    cy = Cycle(T=18, hold=3.8)
    o = csv_("exp5_ablation/results/processed/exp5_ablation_summary.csv")
    n = csv_("exp5_ablation/results/processed/exp5_perstage_ablation_summary.csv")
    go, gn = o[o.variant == "_gain_attribution"].set_index("cell"), n[n.variant == "_gain_attribution"].set_index("cell")
    cells = list(gn.index)
    p = Plate(486, "experiments/exp5_ablation/results/processed/ (exp5_ and exp5_perstage_ ablation_summary) · share of the joint search's gain over the best single dataflow · n = 5 seeds",
              "Bars show the share of the full joint search's gain captured by a restricted search that learns only the dataflow, or only "
              "the tiling. With the original action space dataflow-only search captured nothing; with the per-stage action space it "
              "captures more than the whole.", cycle=cy, title="Dataflow-only search goes from capturing none of the gain to all of it")
    Y = lambda v: lerp(v, 0, 170, 360, 100)
    b = []
    b.append(R(cy, 0.1, yaxis(86, 100, 360, [0, 50, 100, 150], Y, lambda t: f"{t} %", "share of the joint search's gain"), kind="fade"))
    b.append(R(cy, 0.2, line(86, Y(100), 900, Y(100), stroke="ink2", sw=0.9, dash="4 4"), text(96, Y(100) - 6, "joint search = 100 %", 12, "sf", "mute", italic=True), kind="fade"))
    for ci, c in enumerate(cells):
        x0 = 130 + ci * 400
        lab = c.replace(" @ ", " · ").replace("seq", "sequence ")
        b.append(R(cy, 0.2, text(x0 + 150, 84, f"({'ab'[ci]})  {lab}", 14.5, "sf", "ink", anchor="middle"), line(x0 + 10, 366, x0 + 290, 366, stroke="mute", sw=0.9), kind="fade"))
        for vi, (var, tok, nm) in enumerate((("dataflow_only", "warm", "dataflow only"), ("tiling_only", "acc", "tiling only"))):
            vo = float(go.loc[c, f"{var}_pct_of_joint_gain"]); vn = float(gn.loc[c, f"{var}_pct_of_joint_gain"])
            bx_ = x0 + 30 + vi * 140
            b.append(R(cy, 0.6, rect(bx_, Y(vo) if vo > 0 else 359, 46, max(360 - Y(vo), 1.5) if vo > 0 else 1.5, stroke=tok, sw=1.2, dash="4 3"),
                       text(bx_ + 23, (Y(vo) - 6) if vo > 0 else 350, f"{vo:.0f} %", 12, "sf", tok, anchor="middle", italic=True), kind="fade"))
            b.append(rect(bx_ + 52, Y(vn), 46, 360 - Y(vn), fill=tok, op=.88, cls="gy", anim=cy.grow_y(1.6 + vi * 0.5 + ci * 0.3, 1.0)))
            b.append(R(cy, 2.8 + vi * 0.3 + ci * 0.3, text(bx_ + 75, Y(vn) - 7, f"{vn:.0f} %", 14.5, "sf", tok, anchor="middle", weight=700), kind="fade"))
            b.append(text(bx_ + 49, 386, nm, 13, "sf", "ink2", anchor="middle"))
    b.append(R(cy, 3.8, text(130 + 152, 412, "dashed outline = original action space      solid bar = per-stage action space", 12, "sf", "mute", italic=True), kind="fade"))
    b.append(R(cy, 4.6, text(480, 444, "All three variants are statistically indistinguishable (paired tests, every p > 0.23): per-stage dataflow choice was the missing capability.", 13, "sf", "ink2", anchor="middle", italic=True)))
    p.add(*b)
    return p


# ======================================================================================
# Fig. 13 — Experiment 6: zero-shot transfer, seed by seed
# ======================================================================================
def fig13_transfer() -> Plate:
    cy = Cycle(T=18, hold=3.8)
    df = pd.read_csv(EXP / "exp6_generalization/results/raw/exp6_n15_merged.csv")
    df["hd"] = df.hidden_dim // df.num_heads
    agg = csv_("exp6_generalization/results/processed/exp6_n15_agg.csv")
    sig = csv_("exp6_generalization/results/processed/exp6_n15_significance.csv")
    p = Plate(520, "experiments/exp6_generalization/results/raw/exp6_n15_merged.csv · each row a seed · fresh search at 15 iterations",
              "For each of five seeds and two held-out workloads, the utilisation of an action transferred without retraining and of a "
              "freshly searched action. In most seeds the two coincide; gaps appear in two seeds of the different-regime workload and one "
              "of the same-regime workload.", cycle=cy, title="Zero-shot transfer: most seeds coincide with fresh search, a few do not")
    panels = [(48, "(a)", "head dimension 48 · same regime", "D > A: above the knee, like the source", 70.0, 78.0, 150, 420),
              (24, "(b)", "head dimension 24 · different regime", "D < A: below the knee, the opposite regime", 61.0, 70.0, 560, 830)]
    b = []
    totals = {}
    for hd, tag, title, sub, lo, hi, xa, xb in panels:
        X = lambda v, lo=lo, hi=hi, xa=xa, xb=xb: lerp(v, lo, hi, xa, xb)
        c = df[df.hd == hd]
        base = float(c[c.optimizer == "best_fixed_per_stage"].pe_utilization_pct.iloc[0])
        tr = c[c.optimizer == "zero_shot_transfer"].set_index("seed").pe_utilization_pct
        fr = c[c.optimizer == "fresh_ceiling"].copy()
        fr["reward"] = fr.notes.apply(lambda s: json.loads(s)["reward"])
        fr = fr.loc[fr.groupby("seed").reward.idxmax()].set_index("seed").pe_utilization_pct
        total = float((fr - tr).abs().sum()); totals[hd] = total
        head = [text(xa - 40, 78, tag, 14, "sf", "ink", weight=700), text(xa - 18, 78, title, 14.5, "sf", "ink"),
                text(xa - 18, 96, sub, 12, "sf", "mute", italic=True), line(xa - 44, 106, xb + 70, 106, stroke="ink", sw=0.6)]
        b.append(R(cy, 0.1, *head, kind="fade"))
        ax = [line(xa - 10, 360, xb + 10, 360, stroke="mute", sw=0.9)]
        for t in range(int(lo), int(hi) + 1, 2):
            ax += [line(X(t), 360, X(t), 364, stroke="mute", sw=0.9), text(X(t), 380, str(t), 12, "sf", "ink2", anchor="middle")]
        ax.append(text(xb + 10, 400, "PE utilisation, %", 12, "sf", "mute", anchor="end", italic=True))
        ax.append(line(X(base), 124, X(base), 360, stroke="ink2", sw=1, dash="4 4"))
        ax.append(text(X(base), 120, "best per-stage static", 11.5, "sf", "mute", anchor="middle", italic=True))
        b.append(R(cy, 0.3, *ax, kind="fade"))
        for si, s in enumerate(range(5)):
            y = 150 + si * 40
            t_, f_ = float(tr[s]), float(fr[s])
            b.append(R(cy, 0.5, text(xa - 44, y + 4.5, f"seed {s}", 12.5, "sf", "ink2"), line(X(min(t_, f_)), y, X(max(t_, f_)), y, stroke="faint", sw=1.4), kind="fade", dur=0.5))
            b.append(R(cy, 1.0 + si * 0.15, circle(X(t_), y, 5.2, fill="acc"), kind="fade", dur=0.5))
            gap = f_ - t_
            b.append(group(circle(X(f_), y, 5.6, fill="bg", stroke="warm", sw=1.8), anim=cy.slide(3.0 + si * 0.3, X(t_) - X(f_), 0, 1.2) if abs(gap) > 1e-6 else cy.fade(3.0 + si * 0.3, 0.6)))
            b.append(R(cy, 5.0, text(xb + 28, y + 4.5, f"+{gap:.2f}" if gap > 0.005 else "=", 12.5, "sf", "warm" if gap > 0.5 else "mute", italic=True), kind="fade", dur=0.5))
    b.append(R(cy, 1.2, circle(150, 452, 5.2, fill="acc"), text(162, 456, "transferred action, no retraining", 12.5, "sf", "ink2"),
               circle(410, 452, 5.6, fill="bg", stroke="warm", sw=1.8), text(422, 456, "freshly searched (15 iterations)", 12.5, "sf", "ink2")))
    pa, pb = float(sig[sig.cell.str.contains("48") & sig.comparison.str.contains("fresh")].pvalue.iloc[0]), float(sig[sig.cell.str.contains("24") & sig.comparison.str.contains("fresh")].pvalue.iloc[0])
    b.append(R(cy, 5.6, text(480, 488, f"Total gap across seeds: {totals[48]:.2f} pp (same regime) vs. {totals[24]:.2f} pp (different regime) — the predicted direction, but not significant at n = 5 (p = {pa:.2f}, {pb:.2f}).",
                             13, "sf", "ink2", anchor="middle", italic=True)))
    p.add(*b)
    return p


# ======================================================================================
# Fig. 14 — engineering incident: an unused trace file filled the disk
# ======================================================================================
def fig14_incident() -> Plate:
    cy = Cycle(T=16, hold=3.6)
    p = Plate(470, "commit 08e8176 · docker/scalesim/Dockerfile · common/simulator_interface.py (real measurements, same workload and action)",
              "Four steps of an engineering incident and a log-scale chart: a one-line upstream bug made the simulator write about 550 "
              "megabytes of unused trace files per evaluation; after a patch it writes 144 kilobytes.", cycle=cy,
              title="A silent upstream bug filled the disk; a one-line patch removed it")
    steps = [("1", "A sweep fills the disk", "One partial re-run wrote 59 GB of trace files; free space fell to 120 MB."),
             ("2", "The container engine dies", "Out of space mid-run, Docker's daemon crashed and had to be restarted."),
             ("3", "The cause is in the simulator", "Its command line parses a save-trace flag but hard-codes the opposite — so the flag is silently ignored."),
             ("4", "A one-line patch", "The Docker build now corrects that line. The reported metrics are unchanged.")]
    b = []
    for i, (n_, head, body_) in enumerate(steps):
        x = 40 + i * 226
        b.append(R(cy, 0.3 + i * 0.6, text(x, 82, n_, 30, "sf", "acc"), text(x + 26, 78, head, 14.5, "sf", "ink", weight=700),
                   tspans(x + 26, 98, wrap(body_, 33), 12, "sf", "ink2", italic=True, lead=1.4), line(x, 62, x + 200, 62, stroke="ink", sw=0.6)))
    b.append(R(cy, 3.2, text(40, 196, "scale.py", 11.5, "mo", "mute"),
               text(100, 196, "- s = scalesim(save_disk_space=False, …)", 12, "mo", "warm"),
               text(100, 214, "+ s = scalesim(save_disk_space=save_space, …)", 12, "mo", "acc")))
    # log-scale bars
    xa, xb = 250, 880
    lg = lambda kb: lerp(__import__("math").log10(kb), 0, 6, xa, xb)           # 1 KB .. 1 GB
    ticks = [(1, "1 KB"), (100, "100 KB"), (10_000, "10 MB"), (1_000_000, "1 GB")]
    ax = [line(xa, 392, xb, 392, stroke="mute", sw=0.9)] + [line(lg(k), 392, lg(k), 397, stroke="mute", sw=0.9) + text(lg(k), 414, t, 12, "sf", "ink2", anchor="middle") for k, t in ticks]
    b.append(R(cy, 3.0, *ax, text(40, 262, "simulator output written per evaluation", 14.5, "sf", "ink"), kind="fade"))
    before_kb, after_kb = 550 * 1000, 144
    b.append(text(40, 306, "before the patch", 13.5, "sf", "ink2"))
    b.append(rect(xa, 292, lg(before_kb) - xa, 18, fill="warm", op=.85, cls="gx", anim=cy.grow_x(3.4, 1.4)))
    b.append(R(cy, 4.8, text(lg(before_kb) + 10, 306, "≈ 550 MB", 14, "sf", "warm", italic=True), kind="fade"))
    b.append(text(40, 350, "after the patch", 13.5, "sf", "ink2"))
    b.append(rect(xa, 336, lg(after_kb) - xa, 18, fill="acc", cls="gx", anim=cy.grow_x(5.0, 0.8)))
    b.append(R(cy, 5.9, text(lg(after_kb) + 10, 350, "144 KB   ≈ 3,800 × smaller", 14, "sf", "acc", italic=True), kind="fade"))
    b.append(R(cy, 6.6, text(920, 440, "Wall-clock per evaluation also fell, 81 s → 68 s; most of the remaining time is genuine simulation.", 12.5, "sf", "ink2", anchor="end", italic=True)))
    p.add(*b)
    return p


def wrap(s: str, width: int) -> list[str]:
    words, lines_, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines_.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines_.append(cur)
    return lines_


FIGS = {
    "fig00_hero": fig00_hero,
    "fig01_problem": fig01_problem,
    "fig02_dataflows": fig02_dataflows,
    "fig03_loop": fig03_loop,
    "fig04_trajectory": fig04_trajectory,
    "fig05_programme": fig05_programme,
    "fig06_bakeoff": fig06_bakeoff,
    "fig07_knee": fig07_knee,
    "fig08_convergence": fig08_convergence,
    "fig09_redesign": fig09_redesign,
    "fig10_gap": fig10_gap,
    "fig11_scale": fig11_scale,
    "fig12_ablation": fig12_ablation,
    "fig13_transfer": fig13_transfer,
    "fig14_incident": fig14_incident,
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:]
    for name, fn in FIGS.items():
        if wanted and not any(name.startswith(w) for w in wanted):
            continue
        size = write(str(OUT / f"{name}.svg"), fn())
        print(f"{name}.svg  {size / 1024:.1f} KB")
    if svgkit.MISSING_GLYPHS:
        print("glyphs missing from the embedded font (browser will fall back):", "".join(sorted(svgkit.MISSING_GLYPHS)))
