import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('SCEPTRE.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find cells containing key components
keywords = {
    'train_cached': [],
    'class Detector': [],
    'class RSTE': [],
    'BUDGET': [],
    'DEVICE': [],
    'def train_one': [],
    'def evaluate': [],
    'PLANS': [],
    'CTX': [],
    'MODELS': [],
}

for i, c in enumerate(cells):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    for kw in keywords:
        if kw in src:
            keywords[kw].append(i)

for kw, idxs in keywords.items():
    print(f"{kw:20s}: cells {idxs}")
