import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('SCEPTRE.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Show cell 25 (train_cached, evaluate)
c = cells[25]
src = ''.join(c['source'])
print(f"Cell [25] - train_cached, evaluate")
print(src[:3000].encode('ascii', errors='replace').decode('ascii'))
