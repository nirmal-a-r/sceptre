import json

with open('SCEPTRE.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    first_line = src.strip().split('\n')[0][:100] if src.strip() else '(empty)'
    print(f"  [{i:2d}] [{c['cell_type'][:4]}] {first_line}")
