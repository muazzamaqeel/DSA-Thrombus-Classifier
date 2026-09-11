"""Forward-pass timing with resident weights/input and explicit timing scope."""
import statistics
import time

import torch

WARMUP_RUNS = 2
TIMED_RUNS = 3


def run_timed_model(model, image, device, warmup_runs=WARMUP_RUNS, timed_runs=TIMED_RUNS):
    if warmup_runs < 1 or timed_runs < 1:
        raise ValueError("At least one warm-up and one measured forward pass are required.")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    if image.device != device:
        raise ValueError(f"Input is on {image.device}, but {device} was selected.")
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device != device:
            raise ValueError(f"Model tensor is on {tensor.device}, but {device} was selected.")
    model.eval()
    is_cuda = device.type == "cuda"
    samples = []
    with torch.inference_mode():
        if is_cuda:
            # CUDA events initialize lazily. Prime and reuse them outside measured passes.
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(device)
            start.record(stream)
            end.record(stream)
            end.synchronize()
        warm_start = time.perf_counter()
        for _ in range(warmup_runs):
            warm_output = model(image)
            del warm_output
        if is_cuda:
            torch.cuda.synchronize(device)
        warmup_ms = (time.perf_counter() - warm_start) * 1000.0

        for _ in range(timed_runs):
            if is_cuda:
                torch.cuda.synchronize(device)
                wall_start = time.perf_counter()
                start.record(stream)
                logits = model(image)
                end.record(stream)
                end.synchronize()
                host_ms = (time.perf_counter() - wall_start) * 1000.0
                forward_ms = float(start.elapsed_time(end))
            else:
                wall_start = time.perf_counter()
                logits = model(image)
                forward_ms = host_ms = (time.perf_counter() - wall_start) * 1000.0

            # Preserve the existing sigmoid output, outside the timed forward pass.
            activation = float(torch.sigmoid(logits).item())
            del logits
            samples.append({"ForwardMilliseconds": forward_ms,
                            "HostForwardMilliseconds": host_ms,
                            "ModelOutput": activation})

    representative = sorted(samples, key=lambda x: x["ForwardMilliseconds"])[len(samples) // 2]
    return {
        "ModelOutput": representative["ModelOutput"],
        "ForwardMilliseconds": float(statistics.median(x["ForwardMilliseconds"] for x in samples)),
        "HostForwardMilliseconds": float(statistics.median(x["HostForwardMilliseconds"] for x in samples)),
        "WarmupMilliseconds": warmup_ms,
        "WarmupRuns": warmup_runs,
        "TimedRuns": timed_runs,
        "TimingMethod": "CUDA events: median forward" if is_cuda else "CPU perf_counter: median forward",
        "Samples": samples,
    }
