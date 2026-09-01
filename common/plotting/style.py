"""
plotting/style.py — the one shared matplotlib/seaborn style used by every figure in the thesis.

Owns: colour-per-dataflow-mode mapping, colour-per-optimizer mapping (colourblind-safe),
font choices, and the 300 dpi / PNG+SVG output convention (plan §5). Every function in
figures.py imports from here rather than setting its own style, so all figures in the
thesis look like one consistent system.

Palette: Okabe-Ito (Okabe & Ito, 2008) — the standard colourblind-safe categorical
palette, distinguishable under the three common forms of colour-vision deficiency.
DATAFLOW_COLORS uses 4 of its 8 colours (one per dataflow mode); OPTIMIZER_COLORS
uses the remaining 4 for the learned candidates, with a separate neutral grey
reserved for "baseline" so baselines read as visually distinct from every learned
method rather than competing with them for a "colour slot" of their own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

DPI = 300

# Okabe-Ito palette (hex).
_OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

DATAFLOW_COLORS: dict[str, str] = {
    "WS": _OKABE_ITO["blue"],
    "OS": _OKABE_ITO["vermillion"],
    "IS": _OKABE_ITO["bluish_green"],
    "RS": _OKABE_ITO["reddish_purple"],
}

OPTIMIZER_COLORS: dict[str, str] = {
    "reinforce": _OKABE_ITO["orange"],
    "ppo": _OKABE_ITO["sky_blue"],
    "evolutionary": _OKABE_ITO["yellow"],
    "bayesian_opt": _OKABE_ITO["black"],
    "baseline": "#999999",  # neutral grey — deliberately outside the Okabe-Ito set, see module docstring
}


def apply_style() -> None:
    """Apply the shared matplotlib rcParams (fonts, DPI, gridlines, spines, legend).

    Call this once at the top of every function in figures.py before plotting.
    Idempotent — safe to call multiple times.
    """
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "savefig.bbox": "tight",
    })


def save_figure(fig: Any, out_path_stub: Path) -> None:
    """Save a figure as both PNG and SVG at the shared DPI, per plan §5.

    Args:
        fig: A matplotlib Figure.
        out_path_stub: Output path without extension — writes `<out_path_stub>.png`
            and `<out_path_stub>.svg`. `out_path_stub`'s own name is used verbatim
            (string-concatenated, not `Path.with_suffix()`'d) — `with_suffix` treats
            everything after the *last* dot in the name as a replaceable extension,
            which silently mangles any stub containing a literal dot (e.g. a label
            like "exp2_epsilon0.1_..." collapses to "exp2_epsilon0.png", and a second
            call with a different real suffix then overwrites the first — a real bug
            hit and fixed 2026-09-01, not a hypothetical one).
    """
    out_path_stub = Path(out_path_stub)
    out_path_stub.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_stub.with_name(out_path_stub.name + ".png"), dpi=DPI)
    fig.savefig(out_path_stub.with_name(out_path_stub.name + ".svg"))
