"""Record installed research dependencies and perform tiny CPU/CUDA checks.

Run from any working directory with the project's .venv Python. This script
does not import training entrypoints, load workloads, or train a policy.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command_output(command: list[str]) -> dict:
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=20, check=False
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def memory_info() -> dict:
    if os.name != "nt":
        return {}

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_pagefile", ctypes.c_ulonglong),
            ("available_pagefile", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        return {
            "total_physical_bytes": memory.total_physical,
            "available_physical_bytes": memory.available_physical,
        }
    return {"error": "GlobalMemoryStatusEx failed"}


def processor_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor()


def tensor_check(torch, device: str) -> dict:
    try:
        # Exact integer-valued matmul also tests NumPy interop on return to CPU.
        left = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
        right = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2)
        result = left @ right
        expected = torch.tensor(
            [[28.0, 34.0], [76.0, 98.0], [124.0, 162.0]], device=device
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ok = torch.equal(result, expected)
        array_result = result.cpu().numpy().tolist()
        return {"ok": ok, "operation": "float32 (3x4) @ (4x2)", "result": array_result}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def collect() -> dict:
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "isolated_venv": sys.prefix != sys.base_prefix,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": processor_name(),
            "logical_cpu_count": os.cpu_count(),
            "memory": memory_info(),
            "nvidia_smi": command_output([
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,driver_version",
                "--format=csv,noheader",
            ]),
            "gpu_compute_processes": command_output([
                "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ]),
        },
        "dependencies": {},
        "pip_check": command_output([sys.executable, "-m", "pip", "check"]),
        "installed_packages": {
            dist.metadata["Name"]: dist.version
            for dist in sorted(importlib.metadata.distributions(), key=lambda item: item.metadata["Name"].lower())
        },
        "installation": {
            "base_python": r"D:\anaconda\envs\llama_factory\python.exe",
            "commands": [
                r"D:\anaconda\envs\llama_factory\python.exe -m venv .venv",
                r"$env:TEMP=(Resolve-Path '.cache\tmp').Path; $env:TMP=$env:TEMP; .venv\Scripts\python.exe -m ensurepip --upgrade --default-pip",
                r".venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple numpy==1.24.4 pandas==2.0.3 scipy==1.14.1 gym==0.15.4",
                r".venv\Scripts\python.exe -m pip download --no-deps torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128 --dest .cache\wheels",
                r".venv\Scripts\python.exe -u .cache\download_torch.py",
                r".venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple filelock 'typing-extensions>=4.10.0' 'sympy>=1.13.3' networkx jinja2 fsspec",
                r".venv\Scripts\python.exe -m pip install --no-index .cache\wheels\torch-2.7.1+cu128-cp311-cp311-win_amd64.whl",
            ],
            "torch_wheel": {
                "url": "https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp311-cp311-win_amd64.whl",
                "size_bytes": 3273066072,
                "published_sha256": "138c66dcd0ed2f07aafba3ed8b7958e2bed893694990e0b4b55b6b2b4a336aa6",
                "download_method": "8 concurrent resumable 32MiB HTTP ranges from the official PyTorch CDN; final SHA256 verification before installation",
            },
            "recreate": r".venv\Scripts\python.exe -m pip install -r requirements-research.txt",
            "upstream_requirements_preserved": True,
            "deviations": [
                "Selected NumPy 1.24.4 to resolve simultaneous upstream NumPy 1.24.3/1.24.4 pins.",
                "PyTorch changed from 2.1.1 to 2.7.1+cu128 for RTX 5060 Blackwell support.",
                "Pinned transitive Gym OpenCV dependency at 4.11.0.86 to retain NumPy 1.x compatibility.",
            ],
            "sources": [
                "https://pytorch.org/blog/pytorch-2-7/",
                "https://pytorch.org/get-started/previous-versions/",
                "https://pypi.org/project/gym/0.15.4/",
            ],
            "notes": [
                "The venv reuses the locally available Python 3.11.15 interpreter without modifying its global environment.",
                "Windows sandbox temporary-directory writes failed during initial ensurepip; using project-local .cache/tmp resolved this.",
                "Initial pip download redirected to download-r2.pytorch.org and made no progress; switched to the official download.pytorch.org CDN with validated range responses.",
                "CUDA usability is checked with an actual tensor matmul, independently of nvidia-smi's reported CUDA compatibility.",
                "No workload or policy training is launched by this check.",
            ],
        },
    }

    for module_name, distribution_name in [
        ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
        ("gym", "gym"), ("cv2", "opencv-python"), ("torch", "torch"),
    ]:
        try:
            module = importlib.import_module(module_name)
            report["dependencies"][distribution_name] = {
                "version": importlib.metadata.version(distribution_name),
                "module_path": str(module.__file__),
                "import_ok": True,
            }
        except Exception as exc:
            report["dependencies"][distribution_name] = {
                "import_ok": False, "error": f"{type(exc).__name__}: {exc}"
            }

    try:
        import numpy as np
        from gym.utils import seeding
        from scipy.signal import lfilter

        rng, seed = seeding.np_random(7)
        filtered = lfilter([1], [1, -0.5], [1, 0, 0])
        report["simulator_primitives"] = {
            "gym_seed": seed,
            "gym_seeded_sample": int(rng.randint(100)),
            "scipy_lfilter_result": filtered.tolist(),
            "ok": bool(seed == 7 and np.allclose(filtered, [1, 0.5, 0.25])),
        }
    except Exception as exc:
        report["simulator_primitives"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_build_version": torch.version.cuda,
            "cuda_is_available": torch.cuda.is_available(),
            "cpu_matmul": tensor_check(torch, "cpu"),
            "cuda_devices": [],
        }
        if torch.cuda.is_available():
            report["torch"]["cuda_arch_list"] = torch.cuda.get_arch_list()
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                free, total = torch.cuda.mem_get_info(index)
                report["torch"]["cuda_devices"].append({
                    "index": index,
                    "name": properties.name,
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": properties.total_memory,
                    "free_memory_bytes_before_check": free,
                    "cuda_reported_total_memory_bytes": total,
                    "matmul": tensor_check(torch, f"cuda:{index}"),
                })
    except Exception as exc:
        report["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "environment.json")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"environment_report": str(args.output), "python": report["python"],
                      "dependencies": report["dependencies"], "torch": report["torch"]},
                     indent=2, ensure_ascii=False))
    imports_ok = all(value["import_ok"] for value in report["dependencies"].values())
    pip_ok = report["pip_check"].get("returncode") == 0
    torch_report = report["torch"]
    cpu_ok = torch_report.get("cpu_matmul", {}).get("ok", False)
    cuda_ok = bool(torch_report.get("cuda_is_available")) and all(
        device["matmul"]["ok"] for device in torch_report.get("cuda_devices", [])
    )
    primitives_ok = report.get("simulator_primitives", {}).get("ok", False)
    return 0 if pip_ok and imports_ok and primitives_ok and cpu_ok and (cuda_ok or not args.require_cuda) else 1


if __name__ == "__main__":
    raise SystemExit(main())
