```
Prefix	Meaning
No prefix mentioned but not fixed or implemented on that day
*	completed on that day
+	completed on a later day
-	decided against on a later day
~   opinion and musings on that day
```

Find all open tasks
`$ grep '^[^*+-]' .plan`

Find last 5 completed tasks
`$ grep '^\*' .plan | head -5`

= logs ===================================

## ongoing
    DEBUG
    establish project structure through architecture file

## 2026-05-01
    + create nvidia device check and testing environement 

## 2026-05-02
    + set up `Device` class to import device info, launch operations
    + reconfigure forge device check to check through CUDA driver API (ctypes FFI)

## 2026-05-03
    * build `_NvidiaOps`(ops_cuda.py) to handle device requests (orchestrated through `Device`)

## 2026-05-04
    - first pass at matmul 

## 2026-05-05
    * merge `device_info.py` helper functions into Device class

## 2026-05-06
    add metal (apple silicon) support - reference instructions `src/forge/device/support_checklist.md`
        + ops/ops_metal.py
        + create extern c++ function instantiating Metal
        + `_MetalOps` class in `src/forge/ops/ops_metal.py`

## 2026-05-14
    metal kernel for log returns
