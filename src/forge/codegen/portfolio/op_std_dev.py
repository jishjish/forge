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
    if (id == 0 || id >= data_length) return;

    float squared_deviations = 0.0;
    float variance = 0.0;

    for (int a = 0; a < {assets}; a++)
    {{
        float deviation = averages[id] - returns[a * {stride} + id];
        float square = deviation * deviation;
        squared_deviations += square;
    }}

    variance += squared_deviations / {assets};
    float std_dev_res = sqrt(variance);

    std_dev[id] = std_dev_res;
    """