import importlib
import numpy as np
import polars as pl
import pandas as pd
from rich import print
from pathlib import Path
from pydantic import BaseModel
from forge.helpers import DEBUG
from utils import available_portfolio_ops, available_linalg_ops, get_op_import_path, extract_buffer_count_from_source
from shapetracker.shapetracker import ShapeTracker

class Compile:
    _portfolio_ops = available_portfolio_ops()
    _linalg_ops = available_linalg_ops()

    def __init__(self, gpu: BaseModel, import_path: str, op_name: str, decomp_pipeline: list[str]):
        self.gpu = gpu
        self.import_path = import_path
        self.op_name = op_name
        self.decomp_pipeline = decomp_pipeline
        self.output_path = Path(__file__).resolve().parent / "source_code" / f"{op_name}{self.gpu.file_ext}"
        if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] | GPU: {self.gpu.device_type}, op: {self.import_path}, out: {self.output_path}")

    def _build_source(self, op_name: str, **kwargs):
        try: kernel_module = importlib.import_module("forge.codegen.kernel")
        except ModuleNotFoundError: raise RuntimeError(f"Kernel module not found: {self.import_path}")
        kernel_cls = getattr(kernel_module, f"generate_{self.gpu.device_type.lower()}_{op_name}_kernel")
        kernel = kernel_cls(**kwargs)

        op_import_path = get_op_import_path(op_name)
        try: op_module = importlib.import_module(op_import_path)
        except ModuleNotFoundError: raise RuntimeError(f"Op module not found: {op_import_path}")
        op_cls = getattr(op_module, f"generate_{self.gpu.device_type.lower()}")
        ops = op_cls(self.gpu, **kwargs)
        return kernel + ops + "\n}"

    def dispatch(self, source_code, func_name, data, input_buffer, spec_array, buffer_alloc_data, output_shape, realize):
        _ops_file = [file for file in (Path(__file__).parent.parent/"ops").iterdir() if file.stem.startswith("ops_") and file.stem[len("ops_"):].upper() == self.gpu.device_type.upper()]
        module = importlib.import_module(f"forge.ops.{_ops_file[0].stem}")
        cls = getattr(module, f"_{self.gpu.device_type.capitalize()}Compile")
        req = cls()
        if DEBUG >= 1: 
            print(f"[dim]forge ({Path(__file__).name})[/dim] | ops file: {_ops_file}")
            print(f"[dim]forge ({Path(__file__).name})[/dim] | importing module: {module}")
            print(f"[dim]forge ({Path(__file__).name})[/dim] | cls: {cls}")
        return req._compile(source_code, func_name, self.gpu, data, input_buffer, spec_array, buffer_alloc_data, len(buffer_alloc_data), output_shape, realize)     # call compilation file from corresponding device

    def _handle_data(self, gpu: BaseModel, data):
        match type(data):
            case np.ndarray: np_data = data.astype(np.float32)
            case pd.Series | pd.DataFrame: np_data = data.to_numpy(dtype=np.float32)
            case pl.Series | pl.DataFrame: np_data = data.to_numpy().astype(np.float32, copy=False)
            case _ : return "Unsupported data structure type"
        entries = np_data.shape[0]
        assets = np_data.shape[1] if np_data.ndim > 1 else 1
        assert np_data.size % assets == 0, "Improper data dimensions, lengths vary by asset."
        return np_data, {"entries": entries, "assets": assets, "stride": entries}

    def generate(self, data, realize: bool = False):
        ops = [v['op'] for v in self.decomp_pipeline]
        state = {}
        source_code = ""
        cleaned_data = self._handle_data(self.gpu, data)
        data_dim = cleaned_data[1]
        entries_int = int(data_dim["entries"])
        assets_int = int(data_dim["assets"])
        spec_array = np.empty((assets_int, entries_int), dtype=np.float32)
        shp = ShapeTracker(data_dim=data_dim, pipeline=self.decomp_pipeline)
        output_shape = shp.calculate_output_shape()

        for i, step in enumerate(ops):
            if step not in state:
                step_source = self._build_source(op_name=step, **data_dim)
                source_code += step_source
                buffer_alloc_data = extract_buffer_count_from_source(step_source)

                input_data = self.decomp_pipeline[i]['input']
                # realize if True and last operation
                step_realize = realize and (step == ops[-1])

                # check if first run, pass clean data, otherwise state data
                if input_data is None: 
                    res = self.dispatch(source_code, step, cleaned_data[0], None, spec_array, buffer_alloc_data, output_shape, step_realize)
                else: 
                    res = self.dispatch(source_code, step, None, state[input_data], spec_array, buffer_alloc_data, output_shape, step_realize)

                if step_realize: 
                    reshaped_res = res.reshape(output_shape)
                    state[step] = reshaped_res
                else:
                    state[step] = res
        # print(f"state: {state}")
        if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] | Updated source code and wrote to: {self.output_path}")
        return state, spec_array.size, output_shape

if __name__ == "__main__":
    pass