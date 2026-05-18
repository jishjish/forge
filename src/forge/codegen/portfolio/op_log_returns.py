from pydantic import BaseModel


def generate_metal(gpu: BaseModel, **kwargs) -> str:
    return """
    if (id == 0) return;
    returns[id - 1] = log(prices[id] / prices[id - 1]);
    """
