import json
from pathlib import Path

log = json.loads((Path(__file__).parent / "benchmark_log.json").read_text())

for entry in log:
    print(f"\n{entry['ts']}  {entry['op']}")
    for size, v in entry['results'].items():
        print(f"  {size:<18} {v['ratio_avg']:.1f}x")