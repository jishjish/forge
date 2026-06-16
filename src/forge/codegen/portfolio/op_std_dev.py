from pydantic import BaseModel

PIPELINE = [
    {"op": "log_returns", "input": None,                        "type": "elementwise"},
    {"op": "mean",        "input": "log_returns",               "type": "reduction"},
    {"op": "std_dev",     "input": ["log_returns", "mean"],     "type": "elementwise"}
]

def generate_metal(gpu: BaseModel, **kwargs) -> str:
    assets = kwargs.get("assets", 1)
    stride = kwargs.get("stride", kwargs.get("entries", 0))
    return f"""
    float squared_deviations = 0.0;
    for (int t = 0; t < {stride}; t++) 
    {{
        float deviation = averages[id] - returns[t * {assets} + id];
        squared_deviations += deviation * deviation;
    }}
    std_dev[id] = sqrt(squared_deviations / {stride});
    """