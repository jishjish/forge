import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from forge.forge import Forge
from datetime import datetime, timezone

load_dotenv()

VERBOSE = os.getenv("VERBOSE", "0") == "1"
SAVE = "--save" in sys.argv
LOG_PATH = Path(__file__).parent / "benchmark_log.json"

def benchmark(op: str, iters: int = 5):
    _portfolio_ops = [file.stem[len("op_"):] for file in (Path(__file__).parent.parent/"src/forge/codegen/portfolio").iterdir() if file.stem.startswith("op")]
    _linalg_ops = [file.stem[len("op_"):] for file in (Path(__file__).parent.parent/"src/forge/codegen/linalg").iterdir() if file.stem.startswith("op")]
    assert op in _portfolio_ops or op in _linalg_ops, f"{op} not found. Supported ops: {_portfolio_ops + _linalg_ops}"
    data_dict = {"small": 100_000, "med": 500_000, "large": 1_000_000, "xlarge": 2_500_000, "xxlarge": 5_000_000, "xxxlarge": 10_000_000}
    
    f = Forge()
    times = {}
    averages = {}
    for size in data_dict: averages[f"{size}_avg"] = {"numpy_avg_ms": 0.0, "forge_avg_ms": 0.0, "ratio_avg": 0.0}

    for i in range(iters):
        if VERBOSE == 1:
            print(f"\n[ iter {i+1}/{iters} ]")
            print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
        for size, val in data_dict.items():
            data = np.random.uniform(100, 500, size = val).astype(np.float32)
            # numpy
            start_numpy = time.perf_counter()
            np.log(data[1:] / data[:-1])
            numpy_ms = (time.perf_counter() - start_numpy) * 1000
            # forge
            start_forge = time.perf_counter()
            f.run(op, data=data)
            forge_ms = (time.perf_counter() - start_forge) * 1000

            times[f"{size}_{i}"] = {"numpy_ms": numpy_ms, "forge_ms": forge_ms, "ratio": forge_ms / numpy_ms}
            averages[f"{size}_avg"]["numpy_avg_ms"] += numpy_ms
            averages[f"{size}_avg"]["forge_avg_ms"] += forge_ms 
            averages[f"{size}_avg"]["ratio_avg"] += forge_ms / numpy_ms
            ratio = forge_ms / numpy_ms
            if VERBOSE == 1:
                print(f"  {size:<10} {numpy_ms:>9.2f}ms {forge_ms:>9.2f}ms {ratio:>7.1f}x {_ratio_indicator(ratio)}")

    for size in data_dict:
        avg = averages[f"{size}_avg"]
        avg["numpy_avg_ms"] /= iters
        avg["forge_avg_ms"] /= iters
        avg["ratio_avg"] /= iters
    return averages, op 

_R = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"

def _ratio_indicator(ratio: float) -> str:
    if ratio <= 0.5:  return f"{_GREEN}{ratio:>5.1f}x{_R}"
    if ratio <= 1.0:  return f"{_YELLOW}{ratio:>5.1f}x{_R}"
    return f"{_RED}{ratio:>5.1f}x{_R}"

def print_summary(averages: dict, op: str, show_bars: bool = False):
    data_dict = {"small": 100_000, "med": 500_000, "large": 1_000_000, "xlarge": 2_500_000, "xxlarge": 5_000_000, "xxxlarge": 10_000_000}
    
    def _fmt_size(n: int) -> str:
        if n >= 1_000_000: return f"{n // 1_000_000}M"
        return f"{n // 1_000}k"

    max_time = max(v["forge_avg_ms"] for v in averages.values())

    def _bar(ms: float, char: str = "█", width: int = 12) -> str:
        filled = max(1, round((ms / max_time) * width))
        return f"{_DIM}{char * filled}{'░' * (width - filled)}{_R}"

    print(f"\n{_DIM}  {'─'*52}{_R}")
    print(f"\n{_BOLD}  {_CYAN}SUMMARY{_R}  {_DIM}{op} · avg over 5 iters{_R}\n")

    if show_bars:
        print(f"  {_DIM}{'size':<18} {'numpy (ms)':>10} {'forge (ms)':>10} {'ratio':>7}{_R}")
        print(f"  {_DIM}{'─'*18} {'─'*10} {'─'*10} {'─'*7}{_R}")
    else:
        print(f"  {_DIM}{'size':<18} {'numpy (ms)':>12} {'forge (ms)':>12} {'ratio':>7}{_R}")
        print(f"  {_DIM}{'─'*18} {'─'*12} {'─'*12} {'─'*7}{_R}")

    for key, v in averages.items():
        size = key.replace("_avg", "")
        n = data_dict.get(size, 0)
        label = f"{size} ({_fmt_size(n)})"
        ratio = v['ratio_avg']

        if show_bars:
            np_bar = _bar(v['numpy_avg_ms'])
            fg_bar = _bar(v['forge_avg_ms'])
            print(f"  {label:<18} {_DIM}{np_bar}{_R} {_DIM}{np_bar and v['numpy_avg_ms']:>6.2f}ms{_R}")
            print(f"  {'':18} {fg_bar} {v['forge_avg_ms']:>6.2f}ms  {_ratio_indicator(ratio)}")
        else:
            print(f"  {label:<18} {_DIM}{v['numpy_avg_ms']:>12.2f}{_R} {_DIM}{v['forge_avg_ms']:>12.2f}{_R} {_ratio_indicator(ratio)}")

    print(f"\n{_DIM}  {'─'*52}{_R}\n")

def save_results(averages: dict, op: str):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "op": op,
        "op_type": "single",
        "results": averages
    }
    log = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
    log.append(entry)
    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"  {_DIM}saved → benchmark_log.json{_R}")


if __name__ == "__main__":
    averages, op = benchmark("log_returns")
    print_summary(averages, op)
    if SAVE: save_results(averages, op)

