from pydantic import BaseModel

PIPELINE = [
    {"op": "log_returns", "input": None, "type": "elementwise"}
]

def generate_metal(gpu: BaseModel, **kwargs) -> str:
    assets = kwargs.get("assets", 1)
    stride = kwargs.get("stride", kwargs.get("entries", 0))
    return f"""
    if (id == 0 || id >= data_length) return;
    for (int a = 0; a < {assets}; a++)
    {{
        returns[(id - 1) * {assets} + a] = log(prices[id * {assets} + a] / prices[(id - 1) * {assets} + a]);
    }}
    """