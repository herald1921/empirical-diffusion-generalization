# Publication figure typography (2026-10-25)

These exporters provide the current manuscript figures at a final width of
6 inches. Ordinary labels are at least 9.2 pt (9.3 pt for Figure 9); mathematical
subscripts retain their normal relative size. Place the exported PDFs at the
full JMLR text width, without an additional shrink factor.

## Run

Install the repository numerical requirements and `publication/requirements.txt`.
Run from the repository root:

```bash
python publication/resize_figure_labels.py --input /path/to/original_figures --output publication/output
python publication/figure08.py --input /path/to/original_figures/figure_08_synthetic_geometries.pdf
python publication/figure09.py
```

The first command preserves the original vector plots and embedded image streams
in Figures 1-7 and 10, replacing labels at a readable size. Supply the pre-revision
manuscript PDFs separately. It must not be run on its own outputs.
Figure 8 retains the four support rows and five columns, preserves the original
vector geometry panels, and redraws the right-hand curves from the archived
`figure8_audit.json` and topology records. It increases row spacing and uses a
shared legend. Figure 9 uses the frozen `plot_data.npz` and
`field_discrepancies.csv` under `appendix_d_images/figure09/frozen_data/`.
The image-bearing NPZ and manuscript PDF inputs are not included in this public
release; obtain or generate them under the dataset terms described in the main
README. The scripts fail if required inputs are missing.

Figure 9 exports panels (a)-(d) and a continuation containing (e)-(f). Both belong
to Figure 9; the extended-training comparison remains Figure 10. The MNIST
export retains all 56 image arrays and all saved curve/interval values. Display
clipping follows the existing [0,1] convention. This is a typography revision,
not a new training run or a re-estimation of numerical results.

Each script writes a JSON record of its inputs and changes. Existing experiment
entry points, numerical records and historical files are retained unchanged.
