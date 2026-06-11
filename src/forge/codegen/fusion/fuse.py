import re
import importlib
from pathlib import Path
from ...helpers import DEBUG
from ..kernel import KERNELS
from utils import get_op_import_path 

class Fusion:
    def __init__(self, ops, shapes):
        self.decomp_pipeline = ops
        self.ops = [op["op"] for op in ops]
        self.buffers = [buf for op in self.ops for buf in KERNELS[op]["buffers"]]
        self.shapes = shapes

    def _build_kernel(self, segmented_kernels: list[dict]):
        buffers = [buf for op in segmented_kernels for buf in KERNELS[op["op"]]["buffers"]]
        fused_kernel = f"""
        #include <metal_stdlib>
        using namespace metal;
        kernel void {segmented_kernels[-1]["op"]} (\n"""
        seen = {}
        for buffer in buffers:
            if buffer["name"] not in seen:
                seen[buffer["name"]] = buffer
        unique_buffers = list(seen.values())
        for i, buffer in enumerate(unique_buffers):
            line = f"        {buffer['qualifier']} {buffer['type']} {buffer['name']}  [[buffer({i})]],"
            fused_kernel += line + "\n"
        fused_kernel += "        uint id  [[thread_position_in_grid]]\n    ) {"
        return fused_kernel

    def _build_ops(self, gpu, ops, **kwargs):
        op_import_paths = [get_op_import_path(ops)]
        agg_ops = ""
        for path in op_import_paths:
            try: op_module = importlib.import_module(path)
            except ModuleNotFoundError: raise RuntimeError(f"Op module not found: {path}")
            op_cls = getattr(op_module, f"generate_{gpu.device_type.lower()}")
            ops = op_cls(gpu, **kwargs)
            agg_ops += ops
        return agg_ops

    def _localize_variables(self, source_code: str, **kwargs):
        # determine which variables should be local to kernel
        entries = kwargs.get("stride", kwargs.get("entries", 0))
        assets = kwargs.get("assets", 1)
        #TODO: init logic; hardcoded right now for op body if(id...)
        # local_init_values = {"float": {0}, "int": 0}
        local_init_values = {"float": "{0}", "int": "0"}
        final_output = [b["name"] for b in KERNELS[self.ops[-1]]["buffers"] if b["qualifier"] == "device"]
        buffs_to_localize = [op["name"] for op in self.buffers if op["qualifier"] == "device" and op["name"] not in final_output]
        op_body = source_code
        for name in buffs_to_localize:
            is_array = f"{name}[" in source_code
            if is_array:
                continue
            op_body = op_body.replace(name, f"{name}_local")
            op_body = re.sub(rf'.*{name}.*\[\],?\n', '', op_body)
            op_body = re.sub(rf'.*{name}.*\[\[buffer\(\d+\)\]\],?\n', '', op_body)
            match = next((item for item in self.buffers if item["name"] == "returns"), None)
            type_str = match["type"].strip("*&")
            dtype_replacement = local_init_values[type_str]
            op_body = op_body.replace(
                "if (id == 0 || id >= data_length) return;",
                f"if (id == 0 || id >= data_length) return;\n        {type_str} {match["name"]}_local{[assets * entries]} = {dtype_replacement};"
            )
        return op_body
    
    # def _check_buffer_alloc_seq(self, source_code: str):
    #     """ check sequencing of buffers to ensure sequential; buffer(0), buffer(1)..."""
    #     pass

    def can_fuse(self, curr, next):
        same_type = curr["type"] == next["type"]
        same_shape = curr["shape"] == next["shape"]
        local_dep = curr["input"] is None or curr["type"]
        return same_type and same_shape and local_dep

    def _segment_kernels(self):
        merged = [{**d, **s} for d, s in zip(self.decomp_pipeline, self.shapes)]
        segments = [[merged[0]]]
        for i in range(len(merged) - 1):
            if self.can_fuse(merged[i], merged[i + 1]):
                segments[-1].append(merged[i + 1])
            else:
                segments.append([merged[i + 1]])
        return segments

    def fuse(self, gpu, **kwargs):
        fused_code = {}
        segmented_kernels = self._segment_kernels()
        for segment in segmented_kernels:
            kernel_name = "_".join(op["op"] for op in segment)
            header = self._build_kernel(segment)
            body = "\n".join(self._build_ops(gpu, op["op"], **kwargs) for op in segment)
            fused = self._localize_variables(header + body, **kwargs) + "\n}"
            fused_code[kernel_name] = {"source_code": fused}
            if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] | Source after fusion\n: {fused + "\n}"}")
        return fused_code