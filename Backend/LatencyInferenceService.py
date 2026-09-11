import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import nullcontext

import torch

from LatencyDiagnostics import BACKEND_REVISION, get_latency_info
from LatencyTimer import TIMED_RUNS, WARMUP_RUNS, run_timed_model


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class LatencyInferenceService:
    """Isolate setup/transfer cost from warmed-up FP32 forward measurements."""

    def __init__(self, classificator):
        self.classificator = classificator
        self._lock = threading.RLock()
        self._prepared = {}
        self._run_id = None
        self._resident_all = False
        self._environment_json = ""

    def _models(self):
        return list(self.classificator.models_frontal.values()) + list(self.classificator.models_lateral.values())

    def _offload_all(self):
        for model in self._models():
            model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Setup/recovery only; never inside the timed loop.
        self._resident_all = False

    def configure_execution(self, mode, model_folder, device_index=0):
        with self._lock:
            mode = (mode or "").upper()
            if mode not in ("GPU", "CPU"):
                raise ValueError("Execution mode must be GPU or CPU.")
            if not model_folder or not os.path.isdir(model_folder):
                raise ValueError("The selected model folder does not exist.")
            if mode == "GPU":
                if not torch.cuda.is_available():
                    raise ValueError("GPU was selected, but this backend's PyTorch cannot use CUDA. No CPU fallback was used.")
                if not isinstance(device_index, int) or device_index < 0 or device_index >= torch.cuda.device_count():
                    raise ValueError("The selected CUDA device index is unavailable.")
                device = torch.device(f"cuda:{device_index}")
            else:
                device = torch.device("cpu")

            self._run_id = None
            self._prepared.clear()
            self._offload_all()
            self.classificator.preparedImages.clear()
            # CPU checkpoint staging avoids initially allocating all ten models in VRAM.
            self.classificator.load_models(model_folder, device_override=str(device))
            for model in self._models():
                model.float().eval()
                model.device = device  # CnnLstmModel.forward uses this explicit attribute.

            # Same FP32 policy on Turing and newer GPUs; no automatic mixed precision.
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                torch.backends.fp32_precision = "ieee"
                torch.backends.cuda.matmul.fp32_precision = "ieee"
                torch.backends.cudnn.fp32_precision = "ieee"
                torch.backends.cudnn.conv.fp32_precision = "ieee"
                torch.backends.cudnn.rnn.fp32_precision = "ieee"
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = mode == "GPU"

            self._resident_all = False
            if mode == "GPU":
                with torch.cuda.device(device):
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                    weight_bytes = sum(t.numel() * t.element_size() for m in self._models()
                                       for t in list(m.parameters()) + list(m.buffers()))
                    reserve = max(1024 ** 3, int(total_bytes * 0.25))
                    if free_bytes >= weight_bytes + reserve:
                        try:
                            for model in self._models():
                                model.to(device)
                            self._resident_all = True
                        except torch.cuda.OutOfMemoryError:
                            self._offload_all()
            self._run_id = uuid.uuid4().hex
            info = get_latency_info()
            info.update({"Precision": "FP32 (TF32 disabled)", "WarmupRuns": WARMUP_RUNS,
                         "TimedRuns": TIMED_RUNS, "SelectedDevice": str(device),
                         "CudnnBenchmark": torch.backends.cudnn.benchmark,
                         "ModelSha256": {f"{view}/{name}": file_sha256(os.path.join(model_folder, view, name))
                                         for view in ("frontal", "lateral")
                                         for name in sorted(os.listdir(os.path.join(model_folder, view)))
                                         if name.endswith(".pt")}})
            self._environment_json = json.dumps(info, sort_keys=True)
            return self._metadata()

    def _require_run(self, run_id):
        if not self._run_id or run_id != self._run_id:
            raise ValueError("Latency configuration changed or expired. Start a new test run.")

    def _metadata(self):
        device = self.classificator.device
        name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        return {"BackendRevision": BACKEND_REVISION, "RunId": self._run_id,
                "ExecutionProvider": "GPU" if device.type == "cuda" else "CPU",
                "TimingDevice": f"{device} | {name}" if device.type == "cuda" else "cpu",
                "TimingMethod": "CUDA events: median forward" if device.type == "cuda" else "CPU perf_counter: median forward",
                "Precision": "FP32 (TF32 disabled)", "WarmupRuns": WARMUP_RUNS, "TimedRuns": TIMED_RUNS,
                "ModelResidency": "all models on GPU" if self._resident_all else
                                  "one model on GPU" if device.type == "cuda" else "CPU",
                "EnvironmentJson": self._environment_json}

    def prepare_images(self, frontal_path, lateral_path, run_id):
        with self._lock:
            self._require_run(run_id)
            if not all((frontal_path, lateral_path)) or not all(os.path.isfile(p) for p in (frontal_path, lateral_path)):
                raise ValueError("At least one requested image path does not exist.")
            start = time.perf_counter()
            prepared, _ = self.classificator.load_images(frontal_path, lateral_path, False)
            preprocess_ms = (time.perf_counter() - start) * 1000.0
            frontal = torch.unsqueeze(prepared["image"], 0).to(dtype=torch.float32).contiguous()
            lateral = torch.unsqueeze(prepared["imageOtherView"], 0).to(dtype=torch.float32).contiguous()
            details = {"PreprocessMilliseconds": preprocess_ms,
                       "FrontalShape": list(frontal.shape), "LateralShape": list(lateral.shape),
                       "FrontalFileSha256": file_sha256(frontal_path), "LateralFileSha256": file_sha256(lateral_path),
                       "FrontalTensorSha256": tensor_sha256(frontal), "LateralTensorSha256": tensor_sha256(lateral)}
            # One paired case at a time, matching the existing sequential runner.
            self._prepared.clear()
            self._prepared[(frontal_path, lateral_path)] = (frontal, lateral)
            return {"PreparationJson": json.dumps(details, sort_keys=True)}

    def release_images(self, frontal_path, lateral_path, run_id):
        with self._lock:
            if run_id == self._run_id:
                self._prepared.pop((frontal_path, lateral_path), None)

    def _run_view(self, model, cpu_image):
        try:
            return self._run_view_once(model, cpu_image)
        except torch.cuda.OutOfMemoryError:
            if not self._resident_all:
                raise
        # Retry outside the exception handler, after its traceback releases intermediates.
        self._offload_all()
        return self._run_view_once(model, cpu_image)

    def _run_view_once(self, model, cpu_image):
        device = self.classificator.device
        image = None
        with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
            try:
                start = time.perf_counter()
                if not self._resident_all:
                    model.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                model_transfer_ms = (time.perf_counter() - start) * 1000.0
                start = time.perf_counter()
                image = cpu_image.to(device=device, dtype=torch.float32)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                input_transfer_ms = (time.perf_counter() - start) * 1000.0
                result = run_timed_model(model, image, device)
                result.update({"ModelTransferMilliseconds": model_transfer_ms,
                               "InputTransferMilliseconds": input_transfer_ms,
                               "InputShape": list(image.shape)})
                return result
            finally:
                del image
                if not self._resident_all and device.type == "cuda":
                    model.cpu()  # Low-memory policy, outside all recorded forward times.

    def classify(self, model_name, frontal_path, lateral_path, run_id):
        with self._lock:
            self._require_run(run_id)
            if model_name not in self.classificator.models_frontal or model_name not in self.classificator.models_lateral:
                raise ValueError(f"Unknown model: {model_name}")
            images = self._prepared.get((frontal_path, lateral_path))
            if images is None:
                raise ValueError("Images must be prepared before latency classification.")
            start = time.perf_counter()
            frontal = self._run_view(self.classificator.models_frontal[model_name], images[0])
            lateral = self._run_view(self.classificator.models_lateral[model_name], images[1])
            result = self._metadata()
            result.update({"OutputFrontal": [frontal["ModelOutput"]], "OutputLateral": [lateral["ModelOutput"]],
                           "FrontalInferenceMilliseconds": frontal["ForwardMilliseconds"],
                           "LateralInferenceMilliseconds": lateral["ForwardMilliseconds"],
                           "FrontalBenchmarkJson": json.dumps(frontal, sort_keys=True),
                           "LateralBenchmarkJson": json.dumps(lateral, sort_keys=True),
                           "BackendRequestMilliseconds": (time.perf_counter() - start) * 1000.0})
            return result
