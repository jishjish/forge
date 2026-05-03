## CUDA driver API

- Initialize the CUDA driver API: **must be called before any other function from the driver**
    - [cuInit](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html)

- Get the device handle
    - [cuDeviceGet](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE)

- Check latest CUDA version supported by the driver
    - [cuDriverGetVersion](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION)

- Attributes to query
    - [cuDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266)
    - [Attributes](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ge12b8a782bebe21b1ac0091bf9f4e2a3) to pass (*device properties*):
        - cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        - cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        - cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        - cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X/Y/Z)
        - cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X/Y/Z)
        - cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE)
        - cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        - cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)