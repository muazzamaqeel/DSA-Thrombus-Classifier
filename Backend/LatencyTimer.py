import time

import torch


def run_timed_model(classificator, model, image):
    """Time the existing thesis Classificator._run_model execution path.

    The latency feature deliberately does not reimplement model execution.
    Classificator._run_model remains responsible for:
      - moving the selected model to the configured device,
      - calling the original CNN-GRU forward pass,
      - applying sigmoid and extracting the activation,
      - clearing the CUDA cache, and
      - moving the model back to CPU when CUDA is used.

    This wrapper only measures how long that established application path takes.
    """
    if classificator.run_on_cuda:
        # Prevent earlier asynchronous GPU work from leaking into this measurement.
        torch.cuda.synchronize(classificator.device)

    start_time = time.perf_counter()

    # Reuse the ORIGINAL thesis inference function.
    activation, _estimate = classificator._run_model(model, image)

    if classificator.run_on_cuda:
        # Ensure all work triggered by the original execution path is complete.
        torch.cuda.synchronize(classificator.device)

    inference_milliseconds = (time.perf_counter() - start_time) * 1000.0

    return activation, float(inference_milliseconds)
