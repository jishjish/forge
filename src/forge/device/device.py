import ctypes
import platform
import importlib
from pathlib import Path
from dotenv import load_dotenv
from ..constants.device import CHIPS 
from ..helpers import DEBUG
from rich import print

load_dotenv()

class Device:
    def __init__(self):
        self.device_type = self.get_device_type()

    def get_device_type(self):
        system = platform.system()
        if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] | platform: {system}")

        if system == "Darwin": return "Metal"
        # linux paths
        for chip, (lib, init_fn) in CHIPS.items():
            try:
                dll = ctypes.CDLL(lib)
                getattr(dll, init_fn)(0)
                if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] |Detected: {chip}")
                return chip
            except:
                if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] |Skipping: {chip}: {e}")
                continue
        return "UNSUPPORTED"

    def get_device_info(self):
        # Dynamically loads and initializes device ops based on the provided device type. 
        # NOTE: check `src/forge/device/support_checklist.md` for new chip support
        try: 
            # access file from ops if it matches the device type input
            _ops_file = [file for file in (Path(__file__).parent.parent/"ops").iterdir() if file.stem.startswith("ops_") and file.stem[len("ops_"):].upper() == self.device_type.upper()]
            module = importlib.import_module(f"forge.ops.{_ops_file[0].stem}")
            cls = getattr(module, f"_{self.device_type.capitalize()}Ops")
            req = cls()
            if DEBUG >= 1: 
                print(f"[dim]forge ({Path(__file__).name})[/dim] | ops file: {_ops_file}")
                print(f"[dim]forge ({Path(__file__).name})[/dim] | importing module: {module}")
                print(f"[dim]forge ({Path(__file__).name})[/dim] | cls: {cls}")
            return req._device_info()
        except FileNotFoundError:
            raise RuntimeError(f"No file found for device: {self.device_type}.")
        except AttributeError:
            raise RuntimeError(f"_{self.device_type.capitalize()}Ops class not found in module.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize device {self.device_type}: {e}")



if __name__ == "__main__":
    d = Device()
    print(f"Device type: {d.device_type}")
    # d.get_device_info()
    # print(d.supported_devices)

    """ future methods for consideration """
    #     def _get_optimal_block_size(self, op, matrix_dim):
    #         """ calculates best launch params for a given operation"""
    #         pass
    # 
    #     def _supports(self, feature):
    #         """ check compatability flags (eg: tensor core, FP16)"""
    #         pass
    #     
    #     def _benchmark(self):
    #         pass

