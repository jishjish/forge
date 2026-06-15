class ShapeTracker:
    def __init__(self, pipeline: list[dict], data_dim: dict) -> None:
        self.pipeline = pipeline
        self.base_shape = (data_dim["entries"], data_dim["assets"])
        self.shape_states = {}

    def calculate_output_shape(self):
        for step in self.pipeline:
            op = step["op"]
            match op:
                case "log_returns":
                    # shape = (self.base_shape[0], self.base_shape[1] - 1)
                    shape = (self.base_shape[0] - 1, self.base_shape[1])
                case "mean":
                    rows, cols = self.shape_states["log_returns"]["shape"]
                    # shape = (rows, 1)
                    shape = (1, cols)
                case "std_dev":
                    rows, cols = self.shape_states["log_returns"]["shape"]
                    # shape = (rows, 1)
                    shape = (1, cols)
                # case "covariance":
                #     rows, cols = self.shape_states["log_returns"]["shape"]
                #     shape = (rows, rows)
                #     extract = "full"
                case "covariance":
                    rows, cols = self.shape_states["log_returns"]["shape"]
                    shape = (cols, cols)  # assets x assets
                case "matmul":
                    cov_shape = self.shape_states["covariance"]["shape"]
                    std_dev_shape = self.shape_states["std_dev"]["shape"]
                    m, k1 = cov_shape
                    k2, n = std_dev_shape
                    assert k1 == k2
                    shape = (m, n)
                case "normalize":
                    rows, cols = self.shape_states["covariance"]["shape"]
                    shape = (rows, rows)
                case _:
                    return
            self.shape_states[op] = {"op": op, "shape": shape}
        return list(self.shape_states.values())

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