import re
import inspect
from pathlib import Path
import forge.models as _models

def available_gpu_models():
    return [c[1] for c in inspect.getmembers(_models, inspect.isclass) if c[0] != 'BaseModel']

def available_portfolio_ops():
    return [file.stem[len("op_"):] for file in (Path(__file__).parent/"forge/codegen/portfolio").iterdir() if file.stem.startswith("op")]

def available_linalg_ops():
    return [file.stem[len("op_"):] for file in (Path(__file__).parent/"forge/codegen/linalg").iterdir() if file.stem.startswith("op")]

def get_op_import_path(op: str):
    if op in available_portfolio_ops(): import_path = f"forge.codegen.portfolio.op_{op}"
    else: import_path = f"forge.codegen.linalg.op_{op}"
    return import_path

def extract_buffer_count_from_source(source_code: str):
    args = []
    res = {}
    match = re.search(r'kernel\s+void\s+(\w+)\s*\((.*?)\)\s*\{', source_code, re.DOTALL)
    if match: 
        args = match.group(2).split(",")
    for i, param in enumerate(args):
        name = param.split()[-2]
        if "uint" in param:
            continue
        elif "device const" in param:
            # input
            res[i] = {"index": i, "type": "input", "name": name}
        elif "constant" in param:
            # constant 
            res[i] = {"index": i, "type": "constant", "name": name}
        elif "device" in param:
            # output
            res[i] = {"index": i, "type": "output", "name": name}
    return res