# Animated figures

The figures embedded in the root README are generated, not drawn by hand.

- `build_figures.py` — reads the logged results in `experiments/exp*/results/` and writes `figures/fig00…fig14_*.svg`.
- `svgkit.py` — small SVG toolkit: design tokens (light/dark), a looping animation timeline, text/shape primitives, and a font embedder.

```bash
python docs/paper/build_figures.py          # all figures
python docs/paper/build_figures.py fig07    # only figures whose name starts with the prefix
```

Needs `numpy`, `pandas` and `fonttools` (plus `matplotlib`, only to locate the STIX font files). No simulator or Docker required.

Each SVG is self-contained: CSS/SMIL animation only, no scripts or external requests, `prefers-color-scheme` and `prefers-reduced-motion` honoured, and a base-64 WOFF subset of the STIX fonts (SIL OFL 1.1; the subset is renamed "PaperSerif" because the licence reserves the original name — see `STIX-FONT-LICENSE.txt`).
