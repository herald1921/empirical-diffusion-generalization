# Figure 6 time notation

The two horizontal-axis labels now read `Replacement start tau_a`, with tau and its subscript typeset in STIX math. The caption and protocol use the window `[tau_a,tau_b)`, width `tau_b-tau_a=0.30`, and terminal time `tau_end=0.99`.

All numerical settings, curves, images, fonts and colors remain unchanged. Historical code is retained. This publication step follows `publication/terminology_revision_20261106`; the exact input/output PDF hashes are recorded in `manifest.json`.

```text
python update_figure6_time.py INPUT.pdf OUTPUT.pdf
```

Requires PyMuPDF, matplotlib and numpy. The script checks that pixels outside the two axis-label regions are unchanged at 144 dpi. Original PDFs are supplied separately.
