import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('SCEPTRE.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Show first 150 chars of each cell around the transfer area
for i in range(28, 40):
    c = cells[i]
    src = ''.join(c['source'])
    preview = src[:300].encode('ascii', errors='replace').decode('ascii')
    print(f"\n{'='*70}")
    print(f"Cell [{i}] [{c['cell_type']}]")
    print(preview)
