import torch

from LatencyTimer import run_timed_model


class LatencyInferenceService:
    """Latency-only classification logic built around the existing Classificator instance."""

    def __init__(self, classificator):
        self._classificator = classificator

    def classify(self, model_frontal_name, model_lateral_name, image_f, image_l):
        self._validate_request(
            model_frontal_name,
            model_lateral_name,
            image_f,
            image_l)

        image_key = hash(image_f + image_l)
        prepared = self._classificator.preparedImages[image_key]

        images_frontal = torch.unsqueeze(prepared["image"], 0)
        images_lateral = torch.unsqueeze(prepared["imageOtherView"], 0)

        try:
            frontal_activation, frontal_inference_ms = run_timed_model(
                self._classificator,
                self._classificator.models_frontal[model_frontal_name],
                images_frontal)

            lateral_activation, lateral_inference_ms = run_timed_model(
                self._classificator,
                self._classificator.models_lateral[model_lateral_name],
                images_lateral)
        finally:
            del images_frontal
            del images_lateral

        inference_ms = frontal_inference_ms + lateral_inference_ms
        execution_provider = (
            "GPU" if self._classificator.run_on_cuda else "CPU")
        timing_method = (
            "CUDA events + synchronize"
            if self._classificator.run_on_cuda
            else "time.perf_counter")

        return {
            "OutputFrontal": [frontal_activation],
            "OutputLateral": [lateral_activation],
            "FrontalInferenceMilliseconds": float(frontal_inference_ms),
            "LateralInferenceMilliseconds": float(lateral_inference_ms),
            "InferenceMilliseconds": float(inference_ms),
            "TimingDevice": str(self._classificator.device),
            "ExecutionProvider": execution_provider,
            "TimingMethod": timing_method
        }

    def _validate_request(
            self,
            model_frontal_name,
            model_lateral_name,
            image_f,
            image_l):
        if model_frontal_name not in self._classificator.models_frontal:
            raise ValueError(
                f"Unknown frontal model: {model_frontal_name}")

        if model_lateral_name not in self._classificator.models_lateral:
            raise ValueError(
                f"Unknown lateral model: {model_lateral_name}")

        image_key = hash(image_f + image_l)
        if image_key not in self._classificator.preparedImages:
            raise ValueError(
                "Images must be prepared through /AiService/LoadImages "
                "before latency classification.")
