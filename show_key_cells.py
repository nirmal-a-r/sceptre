import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('SCEPTRE.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Show cells 10-20 (data loading, Detector, training setup)
for idx in [10, 19, 20, 23, 25, 26]:
    c = cells[idx]
    src = ''.join(c['source'])
    preview = src[:500].encode('ascii', errors='replace').decode('ascii')
    print(f"\n{'='*70}")
    print(f"Cell [{idx}] [{c['cell_type']}]")
    print(preview)
