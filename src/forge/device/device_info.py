# from models import GPU
import ctypes
from dotenv import load_dotenv
# from constants.device import NVIDIA_CHIPS, SUPPORTED_DEVICES

load_dotenv()

def get_device_type():
    
    chips = {
        "CUDA": ("libcuda.so", "cuInit(0)"),
        "AMD": ("libamdhip64.so", "hipInit(0)"),
        "Metal": ("libMetal.dylib", "MTLCreateSystemDefaultDevice")
    }

    for chip, (lib, init_fn) in chips.items():
        try:
            dll = ctypes.CDLL(lib)
            getattr(dll, init_fn)(0)
            return chip
        except:
            continue
    return "UNSUPPORTED"

           

def get_device_info(device_type: str):
    match device_type:
        case "CUDA":
            pass
        case "AMD":
            pass
        case "ARM64":
            pass
        case _:
            raise RuntimeError(f"Device not supported. Supported GPUs: {SUPPORTED_DEVICES}")

if __name__ == "__main__":
    get_device_type()

