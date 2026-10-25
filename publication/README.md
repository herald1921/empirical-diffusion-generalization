# Reviewed manuscript figures (2026-10-26)

The authors reviewed Figures 1-10 individually. Figures 1-8 and 10 use the
approved label/spacing revisions. Figure 9 retains the original six-panel
composite without splitting, abbreviation, or relabeling. Earlier drafts and
experiment records remain in Git history and their original directories.

## Export the selected versions

Install the numerical requirements and `publication/requirements.txt`. Supply
original, pre-typography manuscript PDFs separately; do not use exported PDFs as
inputs to the label renderer.

```bash
python publication/resize_figure_labels.py --input /path/to/original_figures --output publication/output
python publication/figure08.py --input /path/to/original_figures/figure_08_synthetic_geometries.pdf --output publication/output
python publication/figure09.py --input /path/to/original_figures/figure_09_mnist_comparison.pdf --output publication/output
```

The first command preserves the vector plots and embedded image streams in
Figures 1-7 and 10. Figure 8 preserves the four geometry rows and redraws the
right-hand curves from archived records. Its enlarged geometry panels, wider
curve panels, shared bottom legend, and shared noise-axis label were reviewed
by the authors. Figure 10 includes both `main` and `final` annotations.

Ordinary text in Figures 1-7 and 10 is 9.2 pt at a 6-inch placement width.
Figure 8 is exported at 440 pt width with 9.2 pt ordinary labels (about 9.03 pt
when placed at 6 inches). Mathematical subscripts retain relative sizing.
Figure 9 is the explicitly selected original and is exempt from the enlarged
label treatment; its source PDF is copied byte-for-byte.

Image-bearing NPZ files and manuscript PDFs are not included in the public
release. Obtain or generate these inputs under the dataset terms in the main
README. Missing required inputs cause an error. Historical numerical scripts
and the `run.py` experiment entry points are retained; the commands above are
the current manuscript export workflow.

`reviewed_figures_20261026.json` records the approved artifact hashes. Exporters
write input/selection audits. These are presentation updates, not new training,
evaluation, or estimates of numerical results.
