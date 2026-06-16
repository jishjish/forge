"""
# NOTE: Metal has to be accessed through a C++ shim, because 
Apple provides c++ headers (not a system library). So rather than 
accessing through `CHIPS.get('Metal'), this is invoked through 
`third_party/metal_shim.cpp`. The metal shim file houses 
corresponding functions needed to access device specs
and generate / dispatch kernel functions.
"""
import numpy as np
import ctypes
from rich import print
from pathlib import Path
from ..helpers import DEBUG
from ..models import MetalGPU
from ..constants.device import CHIPS, METAL_ATTRIBUTES, METAL_GPU_FAMILY

class KernelSpecs(ctypes.Structure):
    _fields_ = [
        ("maxThreadgroupMemoryLength", ctypes.c_int),
        ("maxThreadsPerThreadgroupX", ctypes.c_int),
        ("maxThreadsPerThreadgroupY", ctypes.c_int),
        ("maxThreadsPerThreadgroupZ", ctypes.c_int),
        ("recommendedMaxWorkingSetSize", ctypes.c_int)
    ]

class BufferAllocationData(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_int),
        ("type", ctypes.c_char * 16),
        ("name", ctypes.c_char * 32),
    ]

class _MetalOps:
    def __init__(self):
        # ffi through c++
        self.lib = ctypes.CDLL(CHIPS.get("Metal")[0])
        try:
            # set the result types for the c++ shim calls
            self.lib.forge_get_device.restype = ctypes.c_void_p
            self.lib.forge_device_name.restype = ctypes.c_char_p
            self.device = self.lib.forge_get_device()
        except Exception as e: raise RuntimeError(f"Error initializing Metal: {e}")
        self.model = MetalGPU()

    def _device_info(self):
        attributes = {}
        for name, (func, res_type) in METAL_ATTRIBUTES.items():
            if name == "supportsFamily":
                fn = getattr(self.lib, func)
                fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
                fn.restype = ctypes.c_bool
                # iterate backward (most recent device) 
                for i in range(len(METAL_GPU_FAMILY), 0, -1):
                    if fn(self.device, 1000 + i):
                        family_reference = METAL_GPU_FAMILY.get(1000+i)
                        attributes[name] = family_reference
                        break
                continue
            fn = getattr(self.lib, func)
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = getattr(ctypes, res_type)
            result = fn(self.device)
            attributes[name] = result
        if DEBUG >= 1: 
            print(f"[dim]forge ({Path(__file__).name})[/dim] | Metal Specs: {attributes}")
        return MetalGPU(**attributes)
            
class _MetalCompile:
    def __init__(self):
        # ffi through c++
        self.lib = ctypes.CDLL(CHIPS.get("Metal")[0])
        try:
            # set the result types for the c++ shim calls
            self.lib.forge_get_device.restype = ctypes.c_void_p
            self.device = self.lib.forge_get_device()
        except Exception as e: raise RuntimeError(f"Error initializing Metal: {e}")
        self._pipeline_cache = {}
    
    def _compile(
        self, 
        source_code: str, 
        function_name: str, 
        gpu,                                    # pydantic gpu model
        data: np.ndarray | int | list[int],
        spec_array: np.ndarray, 
        buffer_alloc_data,                      # buffer allocation data from kernel
        buffer_alloc_len,
        output_shape,
        realize: bool,
        is_buffer: bool
    ):
        # get function to compile source code; set arg and return type for ffi
        fn = getattr(self.lib, "forge_compile_source")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        fn.restype = ctypes.c_void_p
        result = fn(self.device, source_code.encode("utf-8"), function_name.encode("utf-8"))
        # take lib and generate pipeline
        pipe_fn = getattr(self.lib, "forge_generate_pipeline")
        pipe_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        pipe_fn.restype = ctypes.c_void_p
        # pipeline result returns void, have to unwrap `.value` (memory address)
        if function_name not in self._pipeline_cache:
            pipe_result = ctypes.c_void_p(pipe_fn(self.device, ctypes.c_void_p(result))).value
            self._pipeline_cache[function_name] = pipe_result
        else:
            pipe_result = self._pipeline_cache[function_name]

        # create dispatch pipeline
        dispatch_fn = getattr(self.lib, "forge_dispatch_pipeline") 
        if isinstance(data, np.ndarray):
            data_ptr = [data.ctypes.data]
        elif isinstance(data, list):
            data_ptr = data
        else:
            data_ptr = [data]  # already a void* from previous dispatch

        data_ptr_arr = (ctypes.c_void_p * len(data_ptr))(*data_ptr)
        byte_length = spec_array.nbytes
        data_length = spec_array.shape[0]   # reference entries shape (entries, assets)

        extracted_specs = [key for key in gpu.model_fields.keys() if key in [f[0] for f in KernelSpecs._fields_]]
        kernel_specs = KernelSpecs(**{k: getattr(gpu, k) for k in extracted_specs})
        dispatch_fn.argtypes = [
            ctypes.c_void_p,                        # device
            ctypes.c_void_p,                        # pipeline 
            ctypes.POINTER(KernelSpecs),            # kernel_specs
            ctypes.c_int,                           # data_length 
            ctypes.POINTER(ctypes.c_void_p),        # data_ptr
            ctypes.POINTER(BufferAllocationData),   # buffer allocation data
            ctypes.c_int,                           # buffer allocation length
            ctypes.c_int,                           # byte_length
            ctypes.c_bool                           # buffer boolean
        ]
        dispatch_fn.restype = ctypes.c_void_p
        buffer_alloc_array = (BufferAllocationData * buffer_alloc_len)() # expand buffer to pass entirety
        for i, val in buffer_alloc_data.items():
            buffer_alloc_array[i].index = val["index"]
            buffer_alloc_array[i].type = val["type"].encode("utf-8")
            buffer_alloc_array[i].name = val["name"].encode("utf-8")
        dispatch_res = dispatch_fn(
            self.device, 
            pipe_result, 
            kernel_specs, 
            data_length, 
            data_ptr_arr, 
            buffer_alloc_array, 
            buffer_alloc_len, 
            byte_length, 
            is_buffer,
        )
        if realize:
            read_fn = getattr(self.lib, "forge_read_output_buf")
            read_fn.argtypes = ctypes.c_void_p, ctypes.c_int
            read_fn.restype = ctypes.POINTER(ctypes.c_float)
            data = read_fn(dispatch_res, output_shape[0] * output_shape[1])
            return np.ctypeslib.as_array(data, shape=(output_shape[0], output_shape[1])).copy()
        else:
            return dispatch_res
    
    def realize(self, data_ptr, buffer_shape, output_data_shape):
        read_fn = getattr(self.lib, "forge_read_output_buf")
        read_fn.argtypes = ctypes.c_void_p, ctypes.c_int
        read_fn.restype = ctypes.POINTER(ctypes.c_float)
        data = read_fn(ctypes.c_void_p(data_ptr), buffer_shape[0] * buffer_shape[1])
        return np.ctypeslib.as_array(data, shape=buffer_shape).copy()

if __name__ == "__main__":
    # c = _MetalOps()
    # print(c._device_info())
    pass