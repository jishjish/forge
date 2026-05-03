import ctypes
import importlib
from dotenv import load_dotenv
from forge.constants.device import CHIPS 
from pathlib import Path
# from ops.ops_cuda import _CudaOps

load_dotenv()

def get_device_type():
    for chip, (lib, init_fn) in CHIPS.items():
        try:
            dll = ctypes.CDLL(lib)
            getattr(dll, init_fn)(0)
            return chip
        except:
            continue
    return "UNSUPPORTED"


def get_device_info(device_type: str):
    try: 
        # access file from ops if it matches the device type input
        _ops_file = [file for file in (Path(__file__).parent.parent/"ops").iterdir() if file.stem.startswith("ops_") and file.stem[len("ops_"):].upper() == device_type]
        module = importlib.import_module(f"forge.ops.{_ops_file[0].stem}")
        cls = getattr(module, f"_{device_type.capitalize()}Ops")
        req = cls()
        return req._device_info()
    except FileNotFoundError:
        raise RuntimeError(f"No file found for device: {device_type}.")
    except AttributeError:
        raise RuntimeError(f"_{device_type.capitalize()}Ops class not found in module.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize device {device_type}: e")



    

if __name__ == "__main__":
    # get_device_type()

    print(get_device_info("CUDA"))

    # _ops_file = [file for file in (Path(__file__).parent.parent/"ops").iterdir() if file.stem.startswith("ops_") and file.stem[len("ops_"):].upper() == "CUDA"]
    # print(_ops_file)