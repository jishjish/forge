from pydantic import BaseModel

# log returns
# mean
# dot product
# normalization

PIPELINE = [
    {"op": "log_returns", "input": None},
    {"op": "mean",        "input": "log_returns"},
    {"op": "dot",         "input": "log_returns"},
    {"op": "normalize",   "input": "dot"},
]

def generate_metal(gpu: BaseModel, **kwargs) -> str:
    return """
    if (id == 0 || id >= data_length) return;
    returns[id - 1] = log(prices[id] / prices[id - 1]);
    """