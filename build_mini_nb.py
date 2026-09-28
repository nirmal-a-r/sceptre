"""
Create a minimal notebook with only the cells needed for Cell 22b
(the 118-bus native training experiment), then execute it.

Strategy: include all code cells from index 0 to 33 (Cell 22b), 
skipping the figure-only cells (which are pure matplotlib and produce 
no data needed by later cells). 

However, since the figure cells are interleaved and some variables
(like RESULTS, TRANSFER) depend on earlier code cells, we include
ALL code cells through 33.
"""
import json
import sys
import subprocess
import shutil

sys.stdout.reconfigure(encoding='utf-8')

with open('SCEPTRE.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Identify which code cells produce data needed by Cell 22b
# (vs pure figure cells). We'll skip figure-only cells that
# produce NO globals needed by Cell 22b.
# 
# Figure-only cells (by inspection of content): 4,5,8,9,12,13,14,17,21,26,29,34
# BUT we play it safe: include ALL cells up through 33.

# Build minimal notebook: all cells [0..33]
mini_cells = []
for i, c in enumerate(cells):
    if i <= 33:
        mini_cells.append(c)

mini_nb = {
    "cells": mini_cells,
    "metadata": nb["metadata"],
    "nbformat": nb["nbformat"],
    "nbformat_minor": nb["nbformat_minor"],
}

out_path = "nb_cell22b_run.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(mini_nb, f, ensure_ascii=False, indent=1)

print(f"Created minimal notebook with {len(mini_cells)} cells: {out_path}")
print(f"Original had {len(cells)} cells; kept cells [0..33].")
print()
print("Now run: python -m nbconvert --to notebook --execute --inplace")
print(f"        --ExecutePreprocessor.timeout=7200 {out_path}")
