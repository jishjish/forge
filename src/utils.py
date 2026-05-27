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


if __name__ == "__main__":
    supported_ops = {"portfolio_ops": available_portfolio_ops(), "linalg_ops": available_linalg_ops()}
    print(supported_ops)
