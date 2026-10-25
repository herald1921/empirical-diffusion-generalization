"""Retain the original, single-panel-composite Figure 9 selected by the authors.

The image-bearing original manuscript PDF is supplied separately. This entry
point copies it without modifying labels, panels, images, or numerical curves.
Earlier typography drafts remain available in Git history.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'output')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'figure_09_mnist_comparison.pdf'
    if args.input.resolve() != target.resolve():
        shutil.copyfile(args.input, target)
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    (args.output/'figure09_selection_audit.json').write_text(json.dumps({
        'selection': 'original six-panel composite, unchanged',
        'input_sha256': digest, 'output_sha256': digest,
        'new_simulation': False, 'split_figure': False,
    }, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    main()
