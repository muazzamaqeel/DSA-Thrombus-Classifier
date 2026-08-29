import os

import torch

import Classificator as ClassificatorModule
from LatencyTimer import run_timed_model


class LatencyInferenceService:
    """Latency orchestration around the existing thesis Classificator."""

    def __init__(self, classificator):
        self.classificator = classificator

    def configure_execution(self, mode, model_folder):
        mode = (mode or "").upper()
        if mode not in ("GPU", "CPU"):
            raise ValueError("Execution mode must be GPU or CPU.")
        if not model_folder or not os.path.isdir(model_folder):
            raise ValueError("The selected model folder does not exist.")

        # Reuse the ORIGINAL Classificator.load_models() device logic.
        # CPU_ONLY is changed only while the models are reloaded, then restored.
        previous_cpu_only = ClassificatorModule.CPU_ONLY
        try:
            ClassificatorModule.CPU_ONLY = mode == "CPU"
            self.classificator.preparedImages.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.classificator.load_models(model_folder)
        finally:
            ClassificatorModule.CPU_ONLY = previous_cpu_only

        actual = "GPU" if self.classificator.run_on_cuda else "CPU"
        if mode == "GPU" and actual != "GPU":
            raise ValueError(
                "GPU was selected, but the original classifier could not use CUDA.")

        return {
            "ExecutionProvider": actual,
            "TimingDevice": str(self.classificator.device)
        }

    def prepare_images(self, frontal_path, lateral_path):
        if not all((frontal_path, lateral_path)) or not all(
                os.path.exists(path) for path in (frontal_path, lateral_path)):
            raise ValueError("At least one requested image path does not exist.")

        # Reuse the ORIGINAL thesis preprocessing/cache path.
        self.classificator.prepare_images(frontal_path, lateral_path, False)

    def release_images(self, frontal_path, lateral_path):
        key = hash(frontal_path + lateral_path)
        self.classificator.preparedImages.pop(key, None)

    def classify(self, model_name, frontal_path, lateral_path):
        self._validate(model_name, frontal_path, lateral_path)

        prepared = self.classificator.preparedImages[
            hash(frontal_path + lateral_path)]
        frontal = torch.unsqueeze(prepared["image"], 0)
        lateral = torch.unsqueeze(prepared["imageOtherView"], 0)

        try:
            frontal_output, frontal_ms = run_timed_model(
                self.classificator,
                self.classificator.models_frontal[model_name],
                frontal)
            lateral_output, lateral_ms = run_timed_model(
                self.classificator,
                self.classificator.models_lateral[model_name],
                lateral)
        finally:
            del frontal
            del lateral

        return {
            "OutputFrontal": [frontal_output],
            "OutputLateral": [lateral_output],
            "FrontalInferenceMilliseconds": float(frontal_ms),
            "LateralInferenceMilliseconds": float(lateral_ms),
            "TimingDevice": str(self.classificator.device),
            "ExecutionProvider": "GPU" if self.classificator.run_on_cuda else "CPU"
        }

    def _validate(self, model_name, frontal_path, lateral_path):
        if model_name not in self.classificator.models_frontal:
            raise ValueError(f"Unknown frontal model: {model_name}")
        if model_name not in self.classificator.models_lateral:
            raise ValueError(f"Unknown lateral model: {model_name}")
        if hash(frontal_path + lateral_path) not in self.classificator.preparedImages:
            raise ValueError("Images must be prepared before latency classification.")
