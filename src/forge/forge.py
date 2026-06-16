import uuid
import importlib
from rich import print
from pathlib import Path
from .helpers import DEBUG
from .device.device import Device
from forge.codegen.compile import Compile
from utils import available_gpu_models, available_portfolio_ops, available_linalg_ops, get_op_import_path 

class Forge:
    _gpu_models = available_gpu_models()
    _portfolio_ops, _linalg_ops = available_portfolio_ops(), available_linalg_ops()

    def __init__(self):
        self.device = Device()
        self.gpu_info = None
        self._device_info()
        self._results = {}

    @classmethod
    def supported_gpus(cls): return [m.__name__ for m in cls._gpu_models]

    @classmethod
    def supported_ops(cls): return {"portfolio_ops": cls._portfolio_ops, "linalg_ops": cls._linalg_ops}

    def _device_info(self):
        """ returns corresponding pydantic GPU model; (Forge --> Device --> get_device_info() --> gpu ops)"""
        gpu_info = self.device.get_device_info()
        assert isinstance(gpu_info, tuple(self._gpu_models)), f"Unsupported GPU: {type(gpu_info).__name__}. Supported: {[c.__name__ for c in self._gpu_models]}"
        self.gpu_info = gpu_info
        if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] | [dim]Device initialized:[/dim] {self.gpu_info}")

    def run(self, op: str, realize: bool = False, **kwargs):
        assert op in self._portfolio_ops or op in self._linalg_ops, f"{op} not found. Supported ops: {self._portfolio_ops + self._linalg_ops}"
        import_path = get_op_import_path(op) # import based on provided op (thru portfolio or linalg)

        try: decomp_pipeline = getattr(importlib.import_module(import_path), "PIPELINE")
        except ModuleNotFoundError: raise RuntimeError(f"Error retrieving decomposition pipeline: {import_path}")
        if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] \n | Data realization: {realize} \n | Import path: {import_path} for {op} operation \n | Kwargs: {kwargs}")

        assert self.gpu_info is not None, "Device not initialized"
        data = kwargs.get("data", 1)
        comp = Compile(self.gpu_info, op, decomp_pipeline, data)
        state, data_length, buffer_shape, output_data_shape = comp.generate(realize=realize, **kwargs) 
        handle = str(uuid.uuid4())
        self._results[handle] = {"data_ptr": state, "data_length": data_length, "buffer_shape": buffer_shape, "output_data_shape": output_data_shape["shape"]}
        return handle
    
    def realize(self, handle):
        r = self._results[handle]
        data_ptr, buffer_shape, output_data_shape = r["data_ptr"], r["buffer_shape"], r["output_data_shape"]
        _ops_file = [file for file in (Path(__file__).parent/"ops").iterdir() if file.stem.startswith("ops_") and file.stem[len("ops_"):].upper() == self.gpu_info.device_type.upper()]
        module = importlib.import_module(f"forge.ops.{_ops_file[0].stem}")
        cls = getattr(module, f"_{self.gpu_info.device_type.capitalize()}Compile")
        req = cls()
        return req.realize(data_ptr, buffer_shape, output_data_shape)
       

if __name__ == "__main__":
    import numpy as np

    f = Forge()
    # (entries, assets)
    data=np.random.uniform(100, 500, size=(15, 3)).astype(np.float32)
    data = np.array([
        [100.0, 100.0],
        [110.0, 105.0],
        [121.0, 115.0],
        [133.1, 120.0],
        [146.4, 130.0],
    ], dtype=np.float32)

    data = np.array([
        [99.0, 100.0, 100.0],
        [98.0, 110.0, 105.0],
        [100.0, 121.0, 115.0],
        [102.0, 133.1, 120.0],
        [101.0, 146.4, 130.0],
    ], dtype=np.float32)
    print('data is\n')
    print(data)

    log_ret = f.run('log_returns', data=data)
    fr = f.realize(log_ret)
    print('\n log returns:')
    print(fr)

    mean = f.run('mean', data=data)
    me = f.realize(mean)
    print('\n mean:')
    print(me)

    stddev = f.run('std_dev', data=data)
    st = f.realize(stddev)
    print('\n std dev:')
    print(st)

    
