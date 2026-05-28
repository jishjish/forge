from pydantic import BaseModel

PIPELINE = [
    {"op": "log_returns", "input": None,           "type": "elementwise"},
    {"op": "mean",        "input": "log_returns",  "type": "reduction"},
]

def generate_metal(gpu: BaseModel, **kwargs) -> str:
    assets = kwargs.get("assets", 1)
    stride = kwargs.get("stride", kwargs.get("entries", 0))
    return f"""
    
    float sum = 0;
    int count = {assets};

    for (int a = 0; a < {assets}; a++)
    {{
        sum += returns[a * {stride} + id];
    }}

    float average = sum / count;
    averages[id] = average;
    """
