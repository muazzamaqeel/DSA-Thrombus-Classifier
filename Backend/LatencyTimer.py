import time

import torch


def run_timed_model(classificator, model, image):
    """Measure one execution of the existing Classificator._run_model."""
    if classificator.run_on_cuda:
        torch.cuda.synchronize(classificator.device)

    start_time = time.perf_counter()
    activation, _estimate = classificator._run_model(model, image)

    if classificator.run_on_cuda:
        torch.cuda.synchronize(classificator.device)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return activation, float(elapsed_ms)
