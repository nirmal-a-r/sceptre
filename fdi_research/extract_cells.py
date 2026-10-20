"""
Fix all np.trapezoid -> np.trapz in the notebook (np.trapezoid added in NumPy 2.0,
but system has numpy 1.26.4).
"""
import json, re

nb_path = 'notebooks/FDI_Oracle_Benchmark_v5.ipynb'
nb = json.load(open(nb_path, 'r', encoding='utf-8'))

total_fixes = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    new_src = []
    for line in c['source']:
        fixed = line.replace('np.trapezoid', 'np.trapz')
        if fixed != line:
            print(f"Cell {i}: {line.rstrip()} -> {fixed.rstrip()}")
            total_fixes += 1
        new_src.append(fixed)
    c['source'] = new_src

print(f"\nTotal replacements: {total_fixes}")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Saved.")
