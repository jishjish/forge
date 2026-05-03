import ctypes
import importlib
from pathlib import Path
from dotenv import load_dotenv
from forge.constants.device import CHIPS 

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
    """
    Dynamically loads and initializes device ops based on the provided device type.

    To add support for new chips:
        1. Create supporting `ops_<chip>.py` file in src/forge/ops/ (ex: ops_amd.py)
        2. Ensure the driver function is named `_device_info` in `_<Device>Ops` class
        3. Create a corresponding Pydantic model at src/forge/models/
            a. Update `self.gpu_info` in Forge class to support new model
        4. Return the appropriate GPU model (ex: AMDGPU) from `_device_info()`
        5. Add device lib and init functions to src/constants/device.py
    """
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
        raise RuntimeError(f"Failed to initialize device {device_type}: {e}")


if __name__ == "__main__":
    # get_device_type()

    print(get_device_info("CUDA"))

    # _ops_file = [file for file in (Path(__file__).parent.parent/"ops").iterdir() if file.stem.startswith("ops_") and file.stem[len("ops_"):].upper() == "CUDA"]
    # print(_ops_file)