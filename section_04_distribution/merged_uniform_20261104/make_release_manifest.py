"""List and hash the standalone versioned figure release without copying it."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent
groups={
 'cached_redraw_runtime':['compute_uniform.py','plot_uniform_dimensions.py','curves_dense.csv','dimension_checks.csv','minimum_search.csv','fine_checks.csv','late_stop_checks.csv','late_b_candidates.csv','snapshot_data.npz','protocol_dimensions.txt'],
 'numerical_regeneration':['prepare_uniform_snapshots.py','run_refine.py','run_dimension_checks.py','research_b_selection.py','verify_final.py'],
 'release_instructions':['requirements_uniform.txt','REPRODUCE_UNIFORM.txt','make_release_manifest.py','B_selection_report.txt','protocol.txt'],
 'evidence':['figure_audit.json','dimension_comparison_audit.json','verification.json','symmetry_ot_check.json','dimension_curves.csv','late_b_selection_audit.json'],
 'publication_outputs':['figure_02_stopping_denoising_uniform.pdf','figure_02_stopping_denoising_uniform.png','figure_02_stopping_denoising_uniform.svg'],
}
files=[]
for role,names in groups.items():
    for name in names:
        p=ROOT/name
        if not p.exists():raise FileNotFoundError(p)
        files.append(dict(path=name,role=role,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
manifest=dict(version='uniform_circle_20261104_compact_late_B',figure_inches=[6,5.2],ordinary_font_pt=9.2,B_eta_over_R=.02,right_dimensions=[4,8,16],provenance=dict(repository='https://github.com/herald1921/empirical-diffusion-generalization',commit='5626f1c59d9a9f4951231699ec517afb67779341'),redraw_command='python plot_uniform_dimensions.py --final',cached_redraw_runtime_file_count=len(groups['cached_redraw_runtime']),full_release_file_count=len(files),files=files)
(ROOT/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:manifest[k] for k in ['version','cached_redraw_runtime_file_count','full_release_file_count']},indent=2))
