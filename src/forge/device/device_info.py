import pynvml
from models import GPU
from dotenv import load_dotenv
from constants.device import NVIDIA_CHIPS, SUPPORTED_DEVICES

load_dotenv()

def get_device_type():
    try:
        pynvml.nvmlInit()
        return "CUDA"
    except pynvml.NVMLError: 
        return "UNSUPPORTED"


def get_device_info(device_type: str):
    match device_type:
        case "CUDA":
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)                    # device name

            #TODO: un hard code
            thread_count = 1024

        case "AMD":
            pass
        case "ARM64":
            pass
        case _:
            raise RuntimeError(f"Device not supported. Supported GPUs: {SUPPORTED_DEVICES}")

if __name__ == "__main__":
    get_device_type()
