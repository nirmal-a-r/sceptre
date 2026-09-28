import json
import sys
with open('SCEPTRE.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

with open('inspect_out.txt', 'w', encoding='utf-8') as out:
    for idx in [25, 26, 31, 32, 48, 49]:
        c = cells[idx]
        src = ''.join(c['source'])
        out.write(f'\n{"="*80}\n')
        out.write(f'=== Notebook Cell Index {idx} [{c["cell_type"]}] ===\n')
        out.write('='*80 + '\n')
        out.write(src)
        out.write('\n\n')
print("Done")
