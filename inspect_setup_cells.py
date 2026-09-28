import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('SCEPTRE.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Get the full source of Cell 31 (CELL 22 - zero-shot transfer)
for idx in [1, 3, 4, 5, 7, 8, 9, 31]:
    c = cells[idx]
    src = ''.join(c['source'])
    print(f"\n{'='*70}")
    print(f"Cell [{idx}] [{c['cell_type']}] - first 600 chars")
    print(src[:600].encode('ascii', errors='replace').decode('ascii'))
