"""
svgkit.py — a small, dependency-light toolkit for drawing the README's animated figures.

Aesthetic: a journal figure, not a dashboard. White page, hairline rules, one restrained
accent family, direct labels instead of legends, generous margins, scholarly serif type.

Engineering notes (why the files behave the way they do):

* **Self-contained.** GitHub shows README images through `<img>`, where scripts and external
  files are blocked but inline CSS and SMIL animation run. Everything here is one SVG file.
* **Embedded type.** A README image cannot load web fonts, so each SVG embeds a *subset* of
  STIX General (SIL Open Font License, shipped with matplotlib) as a base-64 WOFF — only the
  glyphs that figure actually uses. Every visitor therefore sees the same typography. If a
  glyph is missing the browser falls back to Palatino/Georgia.
* **Complete picture first.** A figure's *static* state is its finished state. The looping
  animation holds that finished picture for a few seconds, fades, rebuilds element by element,
  and holds again — so static viewers and `prefers-reduced-motion` still get the whole figure.
* **Light/dark.** Colours are CSS custom properties with a `prefers-color-scheme: dark`
  override, and every figure draws its own background so it never depends on the host page.
* **Provenance.** Data figures print the result file their numbers were read from.
"""
from __future__ import annotations

import base64
import html
import io
import re
from pathlib import Path
from typing import Iterable

W = 960  # shared viewBox width so figures stack neatly in the README

SERIF = '"PaperSerif", "Iowan Old Style", "Palatino Linotype", Palatino, Charter, Georgia, "Times New Roman", serif'
MONO = 'ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace'

THEME_CSS = """
:root{--bg:#FFFFFF;--ink:#16181D;--ink2:#3E434C;--mute:#737A85;--faint:#B4B9C1;--hair:#DADDE2;
--wash:#F3F4F6;--acc:#1D4E89;--warm:#B5532B;--pine:#2F6F5E;--gold:#9A7B1E}
@media (prefers-color-scheme:dark){:root{--bg:#0D1117;--ink:#E9ECF1;--ink2:#C4C9D2;--mute:#8C939E;
--faint:#4B525D;--hair:#2A303A;--wash:#151A22;--acc:#7DAAE6;--warm:#E8946F;--pine:#6FC6AA;--gold:#D6B65A}}
.sf{font-family:%(serif)s}.mo{font-family:%(mono)s}
.gy{transform-box:fill-box;transform-origin:50%% 100%%}.gx{transform-box:fill-box;transform-origin:0 50%%}
.pp{transform-box:fill-box;transform-origin:50%% 50%%}.dr{stroke-dasharray:1 2}
@media (prefers-reduced-motion:reduce){*{animation:none !important}}
""" % {"serif": SERIF, "mono": MONO}

ALIAS = {"ws": "acc", "os": "warm", "is": "pine", "paper": "bg", "muted": "mute", "rule": "hair", "grid": "hair",
         "accent": "acc", "soft": "wash", "panel": "wash"}


def esc(s: object) -> str:
    return html.escape(str(s), quote=False)


def col(c: str) -> str:
    """Token ('ink', 'acc', 'ws' …) -> var(--token); literal colours pass through."""
    if c in ("none", "currentColor") or c.startswith(("#", "rgb", "var(", "url(")):
        return c
    c = ALIAS.get(c, c)
    return f"var(--{c})"


def num(x: float) -> str:
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


# --------------------------------------------------------------------------------------------
# rich text: K^{T}, n_{h}, 10^{-4}  ->  tspans with absolute baseline shifts (works in every browser)
_RICH = re.compile(r"(\^\{[^}]*\}|_\{[^}]*\})")


def rich(s: str, size: float) -> str:
    if "^{" not in s and "_{" not in s:
        return esc(s)
    out, restore = [], 0.0
    for tok in _RICH.split(s):
        if not tok:
            continue
        if tok.startswith("^{") or tok.startswith("_{"):
            up = tok[0] == "^"
            shift = (-0.36 if up else 0.2) * size
            out.append(f'<tspan dy="{num(shift - restore)}" font-size="{num(size * 0.72)}">{esc(tok[2:-1])}</tspan>')
            restore = shift
        else:
            out.append(f'<tspan dy="{num(-restore)}">{esc(tok)}</tspan>' if restore else esc(tok))
            restore = 0.0
    return "".join(out)


# --------------------------------------------------------------------------------------------
# font embedding
_FONT_DIR: Path | None = None
_STYLES = {  # (weight, style) -> file
    (400, "normal"): "STIXGeneral.ttf", (400, "italic"): "STIXGeneralItalic.ttf",
    (700, "normal"): "STIXGeneralBol.ttf", (700, "italic"): "STIXGeneralBolIta.ttf",
}
MISSING_GLYPHS: set[str] = set()


def _font_dir() -> Path:
    global _FONT_DIR
    if _FONT_DIR is None:
        import matplotlib
        _FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    return _FONT_DIR


def _subset_woff(path: Path, chars: str) -> bytes:
    from fontTools import subset
    opts = subset.Options()
    opts.flavor = "woff"
    opts.layout_features = ["kern", "liga"]
    opts.name_IDs = [1, 2]
    opts.hinting = False
    opts.notdef_outline = False
    opts.drop_tables += ["DSIG", "GDEF", "FFTM"]
    font = subset.load_font(str(path), opts)
    sub = subset.Subsetter(opts)
    sub.populate(text=chars)
    sub.subset(font)
    # STIX's licence reserves the font name "STIX": a modified (subset) font must not carry it.
    for rec in font["name"].names:
        if rec.nameID in (1, 4, 16):
            rec.string = "PaperSerif"
        elif rec.nameID == 6:
            rec.string = "PaperSerif-Subset"
        elif rec.nameID == 3:
            rec.string = "PaperSerif-Subset-1.0"
    buf = io.BytesIO()
    subset.save_font(font, buf, opts)
    return buf.getvalue()


def font_face_css(svg_body: str) -> str:
    """@font-face rules for the glyphs actually present in the SVG's <text> content."""
    from fontTools.ttLib import TTFont
    texts = re.findall(r"<text\b[^>]*>(.*?)</text>", svg_body, flags=re.S)
    chars = set(html.unescape(re.sub(r"<[^>]+>", "", "".join(texts)))) | set("abcdefghijklmnopqrstuvwxyz0123456789 .,")
    chars.discard("\n")
    css = []
    used_italic = "font-style:italic" in svg_body
    used_bold = 'font-weight="700"' in svg_body
    needed = {(400, "normal")}
    if used_italic:
        needed.add((400, "italic"))
    if used_bold:
        needed.add((700, "normal"))
    if used_italic and used_bold:
        needed.add((700, "italic"))
    for (weight, style), fname in _STYLES.items():
        if (weight, style) not in needed:
            continue
        path = _font_dir() / fname
        cmap = TTFont(str(path)).getBestCmap()
        have = "".join(sorted(c for c in chars if ord(c) in cmap))
        if (weight, style) == (400, "normal"):
            MISSING_GLYPHS.update(c for c in chars if ord(c) not in cmap and not c.isspace())
        b64 = base64.b64encode(_subset_woff(path, have)).decode("ascii")
        css.append(f'@font-face{{font-family:"PaperSerif";font-weight:{weight};font-style:{style};'
                   f'src:url(data:font/woff;base64,{b64}) format("woff")}}')
    return "".join(css)


# --------------------------------------------------------------------------------------------
class Cycle:
    """Loop timeline: hold finished picture -> fade -> rebuild -> hold. Times in seconds."""

    def __init__(self, T: float = 16.0, hold: float = 3.0, fade: float = 0.45):
        self.T, self.hold, self.fade_s = T, hold, fade
        self.t0 = hold + fade + 0.12
        self._kf: dict[tuple, tuple[str, str]] = {}

    def pct(self, t: float) -> float:
        return 100.0 * t / self.T

    def style(self, final: str, init: str, start: float, dur: float = 0.7,
              ease: str = "cubic-bezier(.2,.7,.2,1)") -> str:
        s, e = self.t0 + start, min(self.t0 + start + dur, self.T - 0.2)
        key = (final, init, round(s, 3), round(e, 3), ease)
        if key not in self._kf:
            name = f"k{len(self._kf)}"
            r = self.pct(self.hold + self.fade_s + 0.02)
            body = (f"0%,{num(r)}%{{{final}}}{num(r + 0.01)}%,{num(self.pct(s))}%{{{init};"
                    f"animation-timing-function:{ease}}}{num(self.pct(e))}%,100%{{{final}}}")
            self._kf[key] = (name, f"@keyframes {name}{{{body}}}")
        return f"animation:{self._kf[key][0]} {num(self.T)}s linear infinite"

    def fade(self, start: float, dur: float = 0.7) -> str:
        return self.style("opacity:1", "opacity:0", start, dur)

    def rise(self, start: float, dur: float = 0.7, dy: float = 6) -> str:
        return self.style("opacity:1;transform:translate(0,0)", f"opacity:0;transform:translate(0,{num(dy)}px)", start, dur)

    def grow_y(self, start: float, dur: float = 0.9) -> str:     # class gy
        return self.style("transform:scaleY(1)", "transform:scaleY(0)", start, dur)

    def grow_x(self, start: float, dur: float = 0.9) -> str:     # class gx
        return self.style("transform:scaleX(1)", "transform:scaleX(0)", start, dur)

    def pop(self, start: float, dur: float = 0.5) -> str:        # class pp
        return self.style("opacity:1;transform:scale(1)", "opacity:0;transform:scale(.2)", start, dur,
                          ease="cubic-bezier(.3,1.4,.5,1)")

    def draw(self, start: float, dur: float = 1.3) -> str:       # class dr + pathLength="1"
        return self.style("stroke-dashoffset:0", "stroke-dashoffset:1", start, dur, ease="ease-in-out")

    def slide(self, start: float, dx: float, dy: float = 0, dur: float = 1.1) -> str:
        return self.style("transform:translate(0,0)", f"transform:translate({num(dx)}px,{num(dy)}px)", start, dur,
                          ease="cubic-bezier(.45,.05,.2,1)")

    def css(self) -> str:
        h, f, p = self.hold, self.fade_s, self.pct
        group = (f"@keyframes cyc{{0%,{num(p(h))}%{{opacity:1}}{num(p(h + f))}%,{num(p(h + f + 0.10))}%{{opacity:0}}"
                 f"{num(p(self.t0))}%,100%{{opacity:1}}}}.cyc{{animation:cyc {num(self.T)}s linear infinite}}")
        return group + "".join(b for _, b in self._kf.values())


class Plate:
    """One figure: white page, journal rules top and bottom, content on the cycle, provenance footer."""

    def __init__(self, h: int, source: str, desc: str, title: str = "", cycle: Cycle | None = None,
                 use_cycle: bool = True, rules: bool = True):
        self.h, self.source, self.desc, self.title = h, source, desc, title
        self.cy = cycle or Cycle()
        self.use_cycle, self.rules = use_cycle, rules
        self.body: list[str] = []
        self.extra_css = ""

    def add(self, *parts: str) -> None:
        self.body.extend(parts)

    def render(self) -> str:
        w, h = W, self.h
        content = []
        if self.rules:
            content += [f'<line x1="28" y1="0.75" x2="{w - 28}" y2="0.75" style="stroke:var(--ink);stroke-width:1.5"/>',
                        f'<line x1="28" y1="{h - 0.75}" x2="{w - 28}" y2="{h - 0.75}" style="stroke:var(--ink);stroke-width:1.5"/>']
        open_g = '<g class="cyc">' if self.use_cycle else "<g>"
        content.append(open_g + "".join(self.body) + "</g>")
        if self.source:
            content.append(text(w - 30, h - 14, "Source: " + self.source, 10.5, "sf", "mute", anchor="end", italic=True))
        inner = "".join(content)
        css = THEME_CSS + self.cy.css() + self.extra_css
        head = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
                f'role="img" aria-labelledby="t d"><title id="t">{esc(self.title or self.desc[:80])}</title>'
                f'<desc id="d">{esc(self.desc)}</desc>')
        fonts = font_face_css(inner)
        return f'{head}<style>{fonts}{css}</style><rect width="{w}" height="{h}" style="fill:var(--bg)"/>{inner}</svg>'


# --------------------------------------------------------------------------------------------
# primitives
def text(x: float, y: float, s: str, size: float = 13, fam: str = "sf", fill: str = "ink", *,
         anchor: str = "start", weight: int = 400, italic: bool = False, ls: float = 0,
         anim: str = "", caps: bool = False, op: float = 1, halo: bool = False) -> str:
    st = f"fill:{col(fill)}"
    if halo:
        st += ";paint-order:stroke;stroke:var(--bg);stroke-width:4px;stroke-linejoin:round"
    if italic:
        st += ";font-style:italic"
    if op != 1:
        st += f";opacity:{num(op)}"
    if anim:
        st += ";" + anim
    attrs = f'x="{num(x)}" y="{num(y)}" font-size="{num(size)}" class="{fam}" text-anchor="{anchor}"'
    if weight != 400:
        attrs += f' font-weight="{weight}"'
    if ls or caps:
        attrs += f' letter-spacing="{num(ls or 1.6)}"'
    body = rich(s.upper() if caps else s, size)
    return f'<text {attrs} style="{st}">{body}</text>'


def tspans(x: float, y: float, lines: Iterable[str], size: float = 13, fam: str = "sf", fill: str = "ink", *,
           lead: float = 1.42, anchor: str = "start", weight: int = 400, italic: bool = False, anim: str = "") -> str:
    st = f"fill:{col(fill)}" + (";font-style:italic" if italic else "") + (";" + anim if anim else "")
    spans = "".join(f'<tspan x="{num(x)}" dy="{0 if i == 0 else num(size * lead)}">{rich(l, size)}</tspan>'
                    for i, l in enumerate(lines))
    wt = f' font-weight="{weight}"' if weight != 400 else ""
    return (f'<text x="{num(x)}" y="{num(y)}" font-size="{num(size)}" class="{fam}" text-anchor="{anchor}"{wt} '
            f'style="{st}">{spans}</text>')


def rect(x: float, y: float, w: float, h: float, *, fill: str = "none", stroke: str = "none", sw: float = 1,
         rx: float = 0, op: float = 1, cls: str = "", anim: str = "", dash: str = "") -> str:
    st = f"fill:{col(fill)};stroke:{col(stroke)};stroke-width:{num(sw)}"
    if op != 1:
        st += f";opacity:{num(op)}"
    if dash:
        st += f";stroke-dasharray:{dash}"
    if anim:
        st += ";" + anim
    c = f' class="{cls}"' if cls else ""
    r = f' rx="{num(rx)}"' if rx else ""
    return f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}"{r}{c} style="{st}"/>'


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = "hair", sw: float = 1, dash: str = "",
         op: float = 1, cap: str = "butt", anim: str = "", cls: str = "") -> str:
    st = f"stroke:{col(stroke)};stroke-width:{num(sw)};stroke-linecap:{cap}"
    if dash:
        st += f";stroke-dasharray:{dash}"
    if op != 1:
        st += f";opacity:{num(op)}"
    if anim:
        st += ";" + anim
    c = f' class="{cls}"' if cls else ""
    return f'<line x1="{num(x1)}" y1="{num(y1)}" x2="{num(x2)}" y2="{num(y2)}"{c} style="{st}"/>'


def path(d: str, *, fill: str = "none", stroke: str = "none", sw: float = 1, op: float = 1, cap: str = "round",
         join: str = "round", anim: str = "", draw: bool = False, dash: str = "", cls: str = "") -> str:
    st = f"fill:{col(fill)};stroke:{col(stroke)};stroke-width:{num(sw)};stroke-linecap:{cap};stroke-linejoin:{join}"
    if op != 1:
        st += f";opacity:{num(op)}"
    if dash:
        st += f";stroke-dasharray:{dash}"
    if anim:
        st += ";" + anim
    c = (cls + (" dr" if draw else "")).strip()
    extra = (f' class="{c}"' if c else "") + (' pathLength="1"' if draw else "")
    return f'<path d="{d}"{extra} style="{st}"/>'


def circle(cx: float, cy: float, r: float, *, fill: str = "none", stroke: str = "none", sw: float = 1,
           op: float = 1, anim: str = "", cls: str = "") -> str:
    st = f"fill:{col(fill)};stroke:{col(stroke)};stroke-width:{num(sw)}"
    if op != 1:
        st += f";opacity:{num(op)}"
    if anim:
        st += ";" + anim
    c = f' class="{cls}"' if cls else ""
    return f'<circle cx="{num(cx)}" cy="{num(cy)}" r="{num(r)}"{c} style="{st}"/>'


def group(inner: str, *, anim: str = "", cls: str = "") -> str:
    st = f' style="{anim}"' if anim else ""
    c = f' class="{cls}"' if cls else ""
    return f"<g{c}{st}>{inner}</g>"


def arrow_head(x: float, y: float, ang_deg: float, size: float = 6, fill: str = "ink", op: float = 1) -> str:
    return (f'<path d="M0,0 L{num(-size)},{num(size * 0.42)} L{num(-size)},{num(-size * 0.42)} Z" '
            f'transform="translate({num(x)},{num(y)}) rotate({num(ang_deg)})" style="fill:{col(fill)};opacity:{num(op)}"/>')


def diamond(cx: float, cy: float, r: float, **kw) -> str:
    d = f"M{num(cx)},{num(cy - r)} L{num(cx + r)},{num(cy)} L{num(cx)},{num(cy + r)} L{num(cx - r)},{num(cy)} Z"
    return path(d, **kw)


def glyph(kind: str, cx: float, cy: float, r: float, fill: str, **kw) -> str:
    """Dataflow marks that read without colour: WS square, OS disc, IS diamond."""
    if kind == "ws":
        return rect(cx - r, cy - r, 2 * r, 2 * r, fill=fill, rx=1, **kw)
    if kind == "os":
        return circle(cx, cy, r * 1.1, fill=fill, **kw)
    return diamond(cx, cy, r * 1.3, fill=fill, **kw)


def write(path_: str, plate: Plate) -> int:
    svg = plate.render()
    with open(path_, "w", encoding="utf-8") as f:
        f.write(svg)
    return len(svg.encode("utf-8"))
