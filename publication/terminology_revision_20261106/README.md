# Figure terminology revision, 2026-11-06

This versioned publication step updates labels in the existing vector PDFs without changing their data, curves, images, size, colors, or typography. Historical plotting code and assets remain available.

- Figure 2: `Tweedie` becomes `Denoised` for the two output columns.
- Figure 6: field comparisons use `learned` consistently in row labels, the shared legend and the pure-field baseline label.

Run after the notation revision in `publication/notation_revision_20261106` (or use the exact input PDFs whose SHA-256 hashes are in the manifest):

```text
python update_figure_terminology.py INPUT_FIGURES OUTPUT_FIGURES
```

Dependencies: PyMuPDF, matplotlib and numpy. The script uses matplotlib's STIXGeneral font at the original 9.2-point size. It asserts that all pixels outside the edited label regions are unchanged at 144 dpi. The input PDFs are supplied separately. `terminology_manifest.json` records input/output hashes and individual edits.
