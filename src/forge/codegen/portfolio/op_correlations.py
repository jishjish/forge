from pydantic import BaseModel

PIPELINE = [
    {"op": "log_returns", "input": None,                        "type": "elementwise"},
    {"op": "mean",        "input": "log_returns",               "type": "reduction"},
    {"op": "std_dev",     "input": ["log_returns", "mean"],     "type": "elementwise"},
    {"op": "covariance",  "input": ["log_returns", "mean"],     "type": "reduction"},
    {"op": "matmul",      "input": ["covariance", "std_dev"],   "type": "matmul"},
    {"op": "normalize",   "input": "matmul",                    "type": "elementwise"},
]

def generate_metal(gpu: BaseModel, **kwargs) -> str:
    return """
    if (id == 0 || id >= data_length) return;
    returns[id - 1] = log(prices[id] / prices[id - 1]);
    """