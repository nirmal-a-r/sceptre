import json
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

with open('SCEPTRE.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Extract ALL code cells up to and including cell 33 (22b)
all_code_cells = []
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and i <= 33:
        src = ''.join(c['source'])
        if src.strip():
            all_code_cells.append((i, src))

# Write to a runnable script
with open('run_cell22b_standalone.py', 'w', encoding='utf-8') as out:
    out.write("# Standalone runner: all code cells [0..33] concatenated\n")
    out.write("# This replicates the notebook environment through Cell 22b\n\n")
    for idx, src in all_code_cells:
        out.write(f"\n# {'='*70}\n")
        out.write(f"# === NOTEBOOK CELL [{idx}] ===\n")
        out.write(f"# {'='*70}\n")
        out.write(src)
        out.write("\n")

print(f"Wrote {len(all_code_cells)} code cells to run_cell22b_standalone.py")
total_lines = sum(src.count('\n') for _, src in all_code_cells)
print(f"Total lines: {total_lines}")
