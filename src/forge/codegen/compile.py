import os
import sqlite3
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
        self.output_path = Path(__file__).resolve().parent / "cached_code"
        self._ops_file = [file for file in (Path(__file__).parent.parent/"ops").iterdir() if file.stem.startswith("ops_") and file.stem[len("ops_"):].upper() == self.gpu.device_type.upper()]
        self.cleaned_data = self._handle_data(self.gpu, data)
        self.data_dim = self.cleaned_data[1]
        
        # get shapes throughtout the decomp pipeline
        shp = ShapeTracker(data_dim=self.cleaned_data[1], pipeline=self.decomp_pipeline)
        self.shapes = shp.calculate_output_shape()

        # check for cached code throughout the decomp pipeline
        self.cache = CodeCache(self.gpu, self.output_path)
        self.ops = [op['op'] for op in self.decomp_pipeline]
        self.cached_code = {}
        for op in self.ops: self.cached_code[op] = {"source_code": self.cache.lookup_cache(op_name=op, original_data_shape=str((self.data_dim["entries"], self.data_dim["assets"])), device_name=self.gpu.name)}
        if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] \n | GPU: {self.gpu.device_type}, \n | out: {self.output_path}")

    def dispatch(self, source_code, func_name, data, spec_array, output_shape, realize, is_buffer):
        module = importlib.import_module(f"forge.ops.{self._ops_file[0].stem}")
        cls = getattr(module, f"_{self.gpu.device_type.capitalize()}Compile")
        req = cls()
        buffer_alloc_data = extract_buffer_count_from_source(source_code)
        if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] \n | ops: {self._ops_file} \n | module: {module} \n | cls: {cls}")
        self.cache.write_cache(op_name=func_name, op_code=source_code, original_data_shape=str((self.data_dim["entries"], self.data_dim["assets"])), device_name=self.gpu.name, threadgroup_size=None, execution_ns=None)
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
        # check cache for pre built kernels
        if all(self.cached_code.get(op, {}).get("source_code") for op in self.ops):
            fused_code = self.cached_code
            if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] \n | Cache hit: {fused_code}")
        else:
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




class CodeCache:
    def __init__(self, gpu: BaseModel, output_path: Path):
        self.gpu = gpu
        self.output_path = output_path
        self.db_name = "cached_code.db"
        self.db_path = os.path.join(output_path, self.db_name)
        self.table_name = "cached_ops"
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self._init_cache()

    def _init_cache(self):
        try:
            self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                op_name                     TEXT,                                           -- operation name
                op_code                     TEXT,                                           -- source code
                original_data_shape         TEXT,                                           -- data input shape "(4,3)"
                device_name                 TEXT,                                           -- "Apple M1 Pro"
                threadgroup_size            INTEGER,                         
                execution_ns                INTEGER DEFAULT NULL,                           -- execution in nanoseconds
                PRIMARY KEY                 (op_name, original_data_shape, device_name)
            )
            """)
        except sqlite3.OperationalError as e:
            print(f"Error generating table:\n {e}")

    def lookup_cache(self, op_name: str, original_data_shape: str, device_name: str):
        try: 
            self.cursor.execute(f"""
            SELECT op_code FROM {self.table_name} 
            WHERE op_name = ? AND original_data_shape = ? AND device_name = ?
            """, (op_name, original_data_shape, device_name))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except sqlite3.OperationalError as e:
            print(f"Read error:\n {e}")

    def write_cache(self, op_name, op_code, original_data_shape, device_name, threadgroup_size, execution_ns):
        try: 
            self.cursor.execute(f"""
            INSERT OR REPLACE INTO {self.table_name} (op_name, op_code, original_data_shape, device_name, threadgroup_size, execution_ns)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (op_name, op_code, original_data_shape, device_name, threadgroup_size, execution_ns))
            self.connection.commit()
        except sqlite3.OperationalError as e:
            print(f"Write error:\n {e}")

    def close(self):
        self.connection.close()

    def read_all(self):
        self.cursor.execute(f"SELECT * FROM {self.table_name}")
        return self.cursor.fetchall()



if __name__ == "__main__":
    from src.forge.models import MetalGPU
    gpu = MetalGPU(
        device_type='Metal',
        file_ext='.metal',
        name='Apple M1 Pro',
        maxThreadgroupMemoryLength=32768,
        maxThreadsPerThreadgroupX=1024,
        maxThreadsPerThreadgroupY=1024,
        maxThreadsPerThreadgroupZ=1024,
        recommendedMaxWorkingSetSize=12713115648,
        supportsFamily='apple7'
    )
    output_path = Path('/Users/joshphillips/forge/src/forge/codegen/cached_code')
    cc = CodeCache(gpu=gpu, output_path=output_path)
    # print(cc.read_all())
    print(cc.lookup_cache(op_name="log_returns", original_data_shape="(5, 2)", device_name="Apple M1 Pro"))