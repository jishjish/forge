class ShapeTracker:
    def __init__(self, pipeline: list[dict], data_dim: dict) -> None:
        self.pipeline = pipeline
        self.base_shape = (data_dim["assets"], data_dim["entries"]) # rows / cols
        self.shape_states = {}

    def calculate_output_shape(self):
        for step in self.pipeline:
            op = step["op"]
            match op:
                case "log_returns":
                    shape = self.base_shape
                case "mean":
                    rows, cols = self.shape_states["log_returns"]
                    shape = (rows, 1)
                case "std_dev":
                    rows, cols = self.shape_states["log_returns"]
                    shape = (rows, 1)
                case "covariance":
                    rows, cols = self.shape_states["log_returns"]
                    shape = (rows, rows)
                case "matmul":
                    cov_shape = self.shape_states["covariance"]
                    std_dev_shape = self.shape_states["std_dev"]
                    m, k1 = cov_shape
                    k2, n = std_dev_shape
                    assert k1 == k2
                    shape = (m, n)
                case "normalize":
                    rows, cols = self.shape_states["covariance"]
                    shape = (rows, rows)
                case _:
                    return
            self.shape_states[op] = shape
        return list(self.shape_states.values())[-1]

if __name__ == "__main__":
    PIPELINE = [
        {"op": "log_returns", "input": None,                        "type": "elementwise"},
        {"op": "mean",        "input": "log_returns",               "type": "reduction"},
        {"op": "std_dev",     "input": ["log_returns", "mean"],     "type": "elementwise"},
        {"op": "covariance",  "input": ["log_returns", "mean"],     "type": "reduction"},
        {"op": "matmul",      "input": ["covariance", "std_dev"],   "type": "matmul"},
        {"op": "normalize",   "input": "matmul",                    "type": "elementwise"},
    ]
    data_dim = {"entries": 3, "assets": 9, "stride": 3}
    s = ShapeTracker(PIPELINE, data_dim)

    print(s.calculate_output_shape())
    # s._validate()