# import ctypes
from device_info import get_device_type, get_device_info

class Device:
    def __init__(self):
        self.supported_devices = [get_device_type()]

    def _init_device(self):
        # call for the first device in supported devices
        try: return get_device_info(self.supported_devices[0])
        except: raise RuntimeError("Unsupported device")



if __name__ == "__main__":
    d = Device()
    # print(d.supported_devices)
    print(d._init_device())