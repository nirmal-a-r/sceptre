import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('SCEPTRE.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Full source of cells 31 (CELL 22), 32 (22b markdown), 33 (22b code)
for idx in [31, 33]:
    c = cells[idx]
    src = ''.join(c['source'])
    print(f"\n{'='*70}")
    print(f"Cell [{idx}] [{c['cell_type']}] - FULL SOURCE")
    print('='*70)
    print(src.encode('ascii', errors='replace').decode('ascii'))
