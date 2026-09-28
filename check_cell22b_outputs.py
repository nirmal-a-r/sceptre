import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('SCEPTRE.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
c = cells[33]  # Cell 22b code cell

# Check outputs
outputs = c.get('outputs', [])
print(f"Cell 33 (22b) has {len(outputs)} output(s).")
print(f"execution_count: {c.get('execution_count')}")
if outputs:
    for i, o in enumerate(outputs):
        out_type = o.get('output_type', '')
        text = o.get('text', o.get('data', {}).get('text/plain', ''))
        if isinstance(text, list):
            text = ''.join(text)
        print(f"\n--- Output {i} [{out_type}] ---")
        print(str(text)[:2000].encode('ascii', errors='replace').decode('ascii'))
else:
    print("\nNO OUTPUTS - Cell 22b has NOT been run yet!")
