import importlib
import numpy as np
import polars as pl
import pandas as pd
from rich import print
from pathlib import Path
from pydantic import BaseModel
from forge.helpers import DEBUG
from utils import extract_buffer_count_from_source
from shapetracker.shapetracker import ShapeTracker
from ..codegen.fusion.fuse import Fusion

class Compile:
    def __init__(self, gpu: BaseModel, op_name: str, decomp_pipeline: list[str], data):
        self.gpu = gpu
        self.op_name = op_name
        self.decomp_pipeline = decomp_pipeline
        # TODO: repurpose to cache path for sql lite for kernels
        self.output_path = Path(__file__).resolve().parent / "source_code" / f"{op_name}{self.gpu.file_ext}"
        self._ops_file = [file for file in (Path(__file__).parent.parent/"ops").iterdir() if file.stem.startswith("ops_") and file.stem[len("ops_"):].upper() == self.gpu.device_type.upper()]
        self.cleaned_data = self._handle_data(self.gpu, data)
        self.data_dim = self.cleaned_data[1]
        shp = ShapeTracker(data_dim=self.cleaned_data[1], pipeline=self.decomp_pipeline)
        self.shapes = shp.calculate_output_shape()
        if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] \n | GPU: {self.gpu.device_type}, \n | out: {self.output_path}")

    def dispatch(self, source_code, func_name, data, spec_array, output_shape, realize, is_buffer):
        module = importlib.import_module(f"forge.ops.{self._ops_file[0].stem}")
        cls = getattr(module, f"_{self.gpu.device_type.capitalize()}Compile")
        req = cls()
        buffer_alloc_data = extract_buffer_count_from_source(source_code)
        if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] \n | ops: {self._ops_file} \n | module: {module} \n | cls: {cls}")
        return req._compile(source_code, func_name, self.gpu, data, spec_array, buffer_alloc_data, len(buffer_alloc_data), output_shape, realize, is_buffer)     # call compilation file from corresponding device

    def _handle_data(self, gpu: BaseModel, data):
        match type(data):
            case np.ndarray: np_data = data.astype(np.float32)
            case pd.Series | pd.DataFrame: np_data = data.to_numpy(dtype=np.float32)
            case pl.Series | pl.DataFrame: np_data = data.to_numpy().astype(np.float32, copy=False)
            case _ : return "Unsupported data structure type"
        if np_data.ndim > 1:
            np_data = np.ascontiguousarray(np_data)   # entries-major (rows=entries, cols=assets)
            entries, assets = np_data.shape
        else:
            entries, assets = np_data.shape[0], 1
        stride = entries
        assert np_data.size > 0, "Input array is empty"
        assert np_data.size % assets == 0, "Improper data dimensions, lengths vary by asset."
        return np_data, {"entries": entries, "assets": assets, "stride": stride}

    def _fuse_source_code(self):
        f = Fusion(self.decomp_pipeline, self.shapes)
        return f.fuse(self.gpu, **self.data_dim)
    
    def generate(self, data, realize: bool = False):
        assert self.data_dim["entries"] >= 2, "Calculations require at least two datapoints."
        fused_code = self._fuse_source_code()
        prev_output = self.cleaned_data[0]
        is_buffer = False
        prev_shape = (self.data_dim["entries"], self.data_dim["assets"])
        final_shape = self.shapes[-1]["shape"]

        buffer_cache = {}
        for i, (kernel_name, kernel_data) in enumerate(fused_code.items()):
            output_shape = self.shapes[i]["shape"]
            if (ref := self.decomp_pipeline[i]["input"]) is not None and isinstance(ref, list):
                # multi op inputs, reference first entry
                lookback = self.decomp_pipeline[i]["input"][0]
                prev_shape = next((shape["shape"] for shape in self.shapes if shape["op"] == lookback), None)
            elif ref is not None:
                prev_shape = next((shape["shape"] for shape in self.shapes if shape["op"] == ref), None)
            else:
                prev_shape = prev_shape

            entries_int, assets_int = prev_shape
            # set entries / asset shape as np array
            spec_array = np.empty((entries_int, assets_int), dtype=np.float32)

            if isinstance(self.decomp_pipeline[i]["input"], list):
                inputs = {name: buffer_cache[name] for name in self.decomp_pipeline[i]["input"]}
                res = self.dispatch(kernel_data["source_code"], kernel_name, list(inputs.values()), spec_array, prev_shape, realize, is_buffer)
            else:
                res = self.dispatch(kernel_data["source_code"], kernel_name, prev_output, spec_array, prev_shape, realize, is_buffer)

            buffer_cache[kernel_name] = res
            prev_output = res
            is_buffer = True
            prev_shape = output_shape

        if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] | Updated source code and wrote to: {self.output_path}")
        return res, spec_array.size, final_shape, self.shapes[-1]
