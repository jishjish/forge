"""
https://craftinginterpreters.com/

1) Scanning: take linear stream of chars and chunk them
2) Parsing: takes flat sequence of tokens and builds a tree (parse tree / abstract syntax tree)
3) Static analysis
    - 
"""
import re
import importlib
from ..kernel import KERNELS
from utils import get_op_import_path 

class Fusion:
    def __init__(self, ops):
        self.ops = ops
        self.op_import_paths = [get_op_import_path(op) for op in self.ops]
        self.buffers = [KERNELS[op]["buffers"] for op in self.ops]
        self.flattened_buffers = [item for sublist in self.buffers for item in sublist]

    def _build_kernel(self):
        fused_kernel = f"""
        #include <metal_stdlib>
        using namespace metal;
        kernel void {self.ops[-1]} (\n"""
        seen = {}
        for buffer in self.flattened_buffers:
            if buffer["name"] not in seen:
                seen[buffer["name"]] = buffer
        unique_buffers = list(seen.values())
        for i, buffer in enumerate(unique_buffers):
            line = f"        {buffer['qualifier']} {buffer['type']} {buffer['name']}  [[buffer({i})]],"
            fused_kernel += line + "\n"
        fused_kernel += "        uint id  [[thread_position_in_grid]]\n    ) {"
        return fused_kernel

    def _build_ops(self, gpu, **kwargs):
        agg_ops = ""
        for path in self.op_import_paths:
            try: op_module = importlib.import_module(path)
            except ModuleNotFoundError: raise RuntimeError(f"Op module not found: {path}")
            op_cls = getattr(op_module, f"generate_{gpu.device_type.lower()}")
            ops = op_cls(gpu, **kwargs)
            agg_ops += ops
        return agg_ops

    def _localize_variables(self, source_code: str, **kwargs):
        assets = kwargs.get("assets", 1)
        entries = kwargs.get("stride", kwargs.get("entries", 0))

        local_init_values = {"float": {0}, "int": 0}
        final_output = [b["name"] for b in KERNELS[self.ops[-1]]["buffers"] if b["qualifier"] == "device"]
        buffs_to_localize = [op["name"] for op in self.flattened_buffers if op["qualifier"] == "device" and op["name"] not in final_output]
        op_body = source_code
        for name in buffs_to_localize:
            op_body = op_body.replace(name, f"{name}_local")
            op_body = re.sub(rf'.*{name}.*\[\],?\n', '', op_body)
            op_body = re.sub(rf'.*{name}.*\[\[buffer\(\d+\)\]\],?\n', '', op_body)
            match = next((item for item in self.flattened_buffers if item["name"] == "returns"), None)
            type_str = match["type"].strip("*&")
            dtype_replacement = local_init_values[type_str]
            op_body = op_body.replace(
                "if (id == 0 || id >= data_length) return;",
                f"if (id == 0 || id >= data_length) return;\n        {type_str} {match["name"]}_local{[assets * entries]} = {dtype_replacement};"
            )
        return op_body
    
    def _check_buffer_alloc_seq(self, source_code: str):
        pass

    def fuse(self, gpu, **kwargs):
        base_source_code = self._build_kernel() + self._build_ops(gpu, **kwargs)
        fused = self._localize_variables(base_source_code, **kwargs)
        return fused + "\n}"

if __name__ == "__main__":
    a = """
        #include <metal_stdlib>
            using namespace metal;
            kernel void mean (
        device const float* prices  [[buffer(0)]],
        device float* returns  [[buffer(1)]],
        constant int& data_length  [[buffer(2)]],
        device float* averages  [[buffer(3)]],
        uint id  [[thread_position_in_grid]]
    )
        if (id == 0 || id >= data_length) return;
        for (int a = 0; a < 3; a++)
        {
            int offset = a * 5000;
            if (id % 5000 == 0)
            {
                returns = 0.0;
            } else {
                returns = log(prices / prices);
            }
        }


        float sum = 0;
        int count = 3;

        for (int a = 0; a < 3; a++)
        {
            sum += returns;
        }

        float average = sum / count;
        averages = average;
    """
    ops = ["log_returns", "mean"]

    local_init_values = {"float": 0.0, "int": 0}
    buffers = [KERNELS[op]["buffers"] for op in ops]
    flat_buffs = [item for sublist in buffers for item in sublist]
    final_output = [b["name"] for b in KERNELS[ops[-1]]["buffers"] if b["qualifier"] == "device"]
    buffs_to_localize = [op["name"] for op in flat_buffs if op["qualifier"] == "device" and op["name"] not in final_output]
    op_body = a
    for name in buffs_to_localize:
        op_body = op_body.replace(name, f"{name}_local")
        op_body = re.sub(rf'.*{name}.*\[\],?\n', '', op_body)
        op_body = re.sub(rf'.*{name}.*\[\[buffer\(\d+\)\]\],?\n', '', op_body)
        match = next((item for item in flat_buffs if item["name"] == "returns"), None)
        type_str = match["type"].strip("*&")
        dtype_replacement = local_init_values[type_str]
        op_body = op_body.replace(
            "if (id == 0 || id >= data_length) return;",
            f"if (id == 0 || id >= data_length) return;\n        {type_str} {match["name"]}_local = {dtype_replacement};"
        )
    print(op_body)



    total_elements = 22
    b =f"float {name}_local[{total_elements}] = {{0}};"
    print(b)

