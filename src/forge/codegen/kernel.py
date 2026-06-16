KERNELS = {
    "log_returns": {
        "buffers": [
            {"qualifier": "device const", "type": "float*", "name": "prices"},
            {"qualifier": "device",       "type": "float*", "name": "returns"},
            {"qualifier": "constant",     "type": "int&",   "name": "data_length"},
        ]
    },
    "mean": {
        "buffers": [
            {"qualifier": "device const", "type": "float*", "name": "returns"},
            {"qualifier": "device",       "type": "float*", "name": "averages"},
            {"qualifier": "constant",     "type": "int&",   "name": "data_length"},
        ]
    },
    "std_dev": {
        "buffers": [
            {"qualifier": "device const", "type": "float*", "name": "returns"},
            {"qualifier": "device const",       "type": "float*", "name": "averages"},
            {"qualifier": "device",       "type": "float*", "name": "std_dev"},
            {"qualifier": "constant",     "type": "int&",   "name": "data_length"},
        ]
    },
}