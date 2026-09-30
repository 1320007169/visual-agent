#!/usr/bin/env python3
"""Check the selected GPU in the exact Python environment used by a service."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    import torch

    details = {
        "service": sys.argv[1],
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
    }
    print(json.dumps(details), flush=True)
    try:
        value = torch.ones(1, device="cuda")
        value.add_(1)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        details["error"] = str(exc)
        try:
            driver = ctypes.CDLL("libcuda.so.1")
            version = ctypes.c_int()
            if driver.cuDriverGetVersion(ctypes.byref(version)) == 0:
                details["cuda_driver_api"] = version.value
        except OSError as driver_error:
            details["driver_load_error"] = str(driver_error)
        details["loaded_libcuda"] = sorted({
            line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libcuda.so" in line
        })
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            details["nvidia_smi"] = (result.stdout + result.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as driver_error:
            details["nvidia_smi"] = str(driver_error)
        print(json.dumps(details), file=sys.stderr, flush=True)
        print(
            "CUDA preflight failed before starting services. Check the node driver and loaded libcuda; "
            "use a compatible ModelArts runtime/node if the host driver is too old. "
            "Changing CUDA_HOME alone does not upgrade the driver.",
            file=sys.stderr,
        )
        return 2
    print(f"CUDA preflight passed: {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
