import time

import flask
import torch
from flask import Blueprint, request


def create_latency_blueprint(classificator):
    """Create latency-only routes without changing the normal classification API."""
    blueprint = Blueprint("latency_test", __name__)

    @blueprint.route("/AiService/LatencyClassification", methods=["POST"])
    def latency_classification_requested():
        content = request.get_json()
        model_frontal_name = content["ModelFrontal"]
        model_lateral_name = content["ModelLateral"]
        image_f = content["PathFrontal"]
        image_l = content["PathLateral"]

        if model_frontal_name not in classificator.models_frontal:
            return flask.Response(
                f"Unknown frontal model: {model_frontal_name}",
                status=400)

        if model_lateral_name not in classificator.models_lateral:
            return flask.Response(
                f"Unknown lateral model: {model_lateral_name}",
                status=400)

        image_key = hash(image_f + image_l)
        if image_key not in classificator.preparedImages:
            return flask.Response(
                "Images must be prepared through /AiService/LoadImages before latency classification.",
                status=400)

        prepared = classificator.preparedImages[image_key]
        images_frontal = torch.unsqueeze(prepared['image'], 0)
        images_lateral = torch.unsqueeze(prepared['imageOtherView'], 0)

        try:
            frontal_activation, frontal_inference_ms = _run_timed_model(
                classificator,
                classificator.models_frontal[model_frontal_name],
                images_frontal)

            lateral_activation, lateral_inference_ms = _run_timed_model(
                classificator,
                classificator.models_lateral[model_lateral_name],
                images_lateral)
        finally:
            del images_frontal
            del images_lateral

        inference_ms = frontal_inference_ms + lateral_inference_ms
        execution_provider = "GPU" if classificator.run_on_cuda else "CPU"
        timing_method = (
            "CUDA events + synchronize"
            if classificator.run_on_cuda
            else "time.perf_counter")

        result = {
            'OutputFrontal': [frontal_activation],
            'OutputLateral': [lateral_activation],
            'FrontalInferenceMilliseconds': float(frontal_inference_ms),
            'LateralInferenceMilliseconds': float(lateral_inference_ms),
            'InferenceMilliseconds': float(inference_ms),
            'TimingDevice': str(classificator.device),
            'ExecutionProvider': execution_provider,
            'TimingMethod': timing_method
        }

        print(
            f"Latency classification done. Results: {[frontal_activation]} | {[lateral_activation]}. "
            f"Inference timing: frontal={frontal_inference_ms:.3f} ms, "
            f"lateral={lateral_inference_ms:.3f} ms, "
            f"total={inference_ms:.3f} ms, "
            f"execution={execution_provider}, device={classificator.device}, "
            f"method={timing_method}")

        return flask.jsonify(result)

    return blueprint


@torch.no_grad()
def _run_timed_model(classificator, model, image):
    """Run one model using the same execution semantics as Classificator._run_model.

    Only model execution plus sigmoid activation is timed. Model transfer to/from
    the GPU remains outside the measured interval, matching the current latency
    implementation.
    """
    if classificator.run_on_cuda:
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
