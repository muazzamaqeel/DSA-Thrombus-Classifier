
"""Read-only comparison metadata; unavailable driver details are reported explicitly."""
import os
import platform
import subprocess

import torch
from LatencyRuntimeLimits import available_ram, CNN_GROUPS_PER_BATCH

BACKEND_REVISION = "latency-v7-gpu-batched-single-pass"


def get_latency_info():
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({"Index": index, "Name": properties.name,
                            "TotalVramGiB": properties.total_memory / 1073741824,
                            "ComputeCapability": f"{properties.major}.{properties.minor}"})
    fields = ("index,name,uuid,driver_version,pstate,temperature.gpu,utilization.gpu,"
              "memory.used,memory.total,power.draw,power.limit,clocks.current.sm,clocks.current.memory")
    try:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv"],
                                capture_output=True, text=True, timeout=5, check=False)
        gpu_state = result.stdout.strip() if result.returncode == 0 else "Unavailable: " + result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu_state = f"Unavailable: {error}"
    return {
        "BackendRevision": BACKEND_REVISION,
        "AvailableRamGiB": available_ram() / 1073741824,
        "CnnGroupsPerBatch": CNN_GROUPS_PER_BATCH,
        "Pid": os.getpid(),
        "TorchCudaArchitectures": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "Devices": devices,
        "TorchVersion": str(torch.__version__),
        "CudaVersion": str(torch.version.cuda),
        "CudnnVersion": str(torch.backends.cudnn.version()),
        "Cpu": platform.processor(), "CpuThreads": torch.get_num_threads(),
        "Platform": platform.platform(), "PythonVersion": platform.python_version(),
        "CudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "CudaLaunchBlocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "not set"),
        "NvidiaSmiSnapshot": gpu_state,
    }
