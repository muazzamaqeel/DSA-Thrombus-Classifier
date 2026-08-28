import time

import torch


@torch.no_grad()
def run_timed_model(classificator, model, image):
    """Measure one model execution without changing the normal Classificator code."""
    if classificator.run_on_cuda:
        # Model movement is intentionally outside the scientific inference timer.
        model.to(classificator.device)

        torch.cuda.synchronize(classificator.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output = model(image)
        activation_tensor = torch.sigmoid(output)
        end_event.record()

        torch.cuda.synchronize(classificator.device)
        inference_milliseconds = start_event.elapsed_time(end_event)
        activation = activation_tensor.item()
    else:
        start_time = time.perf_counter()
        output = model(image)
        activation_tensor = torch.sigmoid(output)
        activation = activation_tensor.item()
        inference_milliseconds = (time.perf_counter() - start_time) * 1000.0

    del output
    del activation_tensor
    torch.cuda.empty_cache()

    if classificator.run_on_cuda:
        model.cpu()

    return activation, float(inference_milliseconds)
