# Manuscript notation update, 2026-11-06

Run python update_figure_notation.py INPUT_FIGURES OUTPUT_FIGURES after generating the archived publication figures. Requires PyMuPDF, matplotlib and numpy. Input PDFs are supplied separately; no image data are included here.

Figures 2 and 7: the normalized empirical reference is labelled 1. Figures 4, 8 and 9: paired datasets use calligraphic S with subscripts 1 and 2. Curve data, raw/denoised labels, time parameters, dimensions, colours and font sizes are preserved. The script checks expected replacement counts and verifies zero changed pixels outside label regions at 144 dpi. The manifest records exact input/output hashes.

All historical plotting code remains unchanged. This postprocessing step is the current reproducible notation layer.
