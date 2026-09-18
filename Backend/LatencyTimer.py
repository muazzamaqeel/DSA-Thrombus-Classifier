"""One synchronized prediction per model/view, with a separate forward metric."""
import time
import torch

WARMUP_RUNS = 1  # Once per checkpoint/view per test run, on its first case only.
TIMED_RUNS = 1


def run_timed_model(model, image, device, warmup_runs=WARMUP_RUNS,
                    timed_runs=TIMED_RUNS, cancel_check=lambda: None):
    if warmup_runs not in (0, 1) or timed_runs != 1:
        raise ValueError("This protocol permits zero/one warm-up and exactly one measured pass.")
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    if image.device != device:
        raise ValueError(f"Input device {image.device} does not match {device}.")
    if any(t.device != device for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Model tensors are not all on the selected device.")
    model.eval()
    is_cuda = device.type == 'cuda'
    with torch.inference_mode():
        if is_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(device)
            start.record(stream)
            end.record(stream)
            end.synchronize()
        warm_start = time.perf_counter()
        for _ in range(warmup_runs):
            cancel_check()
            output = model(image, cancel_check=cancel_check)
            del output
        if is_cuda:
            torch.cuda.synchronize(device)
        warm_ms = (time.perf_counter() - warm_start) * 1000.0 if warmup_runs else 0.0
        cancel_check()
        wall_start = time.perf_counter()
        if is_cuda:
            start.record(stream)
        logits = model(image, cancel_check=cancel_check)
        if is_cuda:
            end.record(stream)
            end.synchronize()
        host_forward_ms = (time.perf_counter() - wall_start) * 1000.0
        # Classification ends only when the sigmoid probability is available on the CPU.
        activation = float(torch.sigmoid(logits).item())
        classification_ms = (time.perf_counter() - wall_start) * 1000.0
        forward_ms = float(start.elapsed_time(end)) if is_cuda else host_forward_ms
        del logits
        cancel_check()
    if not 0.0 <= activation <= 1.0:
        raise ValueError("Model produced a non-finite/invalid probability; result rejected.")
    sample = {"ForwardMilliseconds": forward_ms, "HostForwardMilliseconds": host_forward_ms,
              "ClassificationMilliseconds": classification_ms, "ModelOutput": activation}
    return {**sample, "WarmupMilliseconds": warm_ms, "WarmupRuns": warmup_runs, "TimedRuns": 1,
            "TimingMethod": "synchronized wall clock; CUDA stream forward diagnostic" if is_cuda else "CPU wall clock",
            "Samples": [sample]}
