"""Latency-owned model state. Original preprocessing is called without normal caches."""
import gc
import os
import threading
import nibabel
import torch
from Classificator import Classificator
from LatencyCnnLstmModel import LatencyCnnLstmModel
from LatencyRuntimeLimits import require_ram

class LatencyRuntime:
    def __init__(self):
        self.execution_lock = threading.RLock()
        self._active_model = self._active_key = None
        self.models_frontal, self.models_lateral = {}, {}
        self.models_loaded = {'f': '', 'l': ''}
        self.preparedImages = {}
        self.device = torch.device('cpu')
        self.run_on_cuda = False

    def release_model(self):
        self._active_model = None
        self._active_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_models(self, folder="models", device_override=None):
        # Register paths only: selecting a folder must not allocate ten networks.
        paths = {}
        for view in ("frontal", "lateral"):
            directory = os.path.join(folder, view)
            if not os.path.isdir(directory):
                raise ValueError(f"Missing model directory: {directory}")
            paths[view] = {name: os.path.join(directory, name)
                           for name in sorted(os.listdir(directory)) if name.endswith(".pt")}
        if not paths["frontal"] or paths["frontal"].keys() != paths["lateral"].keys():
            raise ValueError("Frontal and lateral checkpoint filenames must match.")
        device = torch.device(device_override) if device_override is not None else torch.device(
            "cuda:0" if torch.cuda.is_available()  else "cpu")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable; no CPU fallback was used.")
            index = device.index if device.index is not None else 0
            if not 0 <= index < torch.cuda.device_count():
                raise ValueError("Selected CUDA device is unavailable.")
            device = torch.device(f"cuda:{index}")
            torch.cuda.set_device(device)
        elif device.type != "cpu":
            raise ValueError("Only CPU and CUDA devices are supported.")
        self.release_model()
        self.device = device
        self.run_on_cuda = device.type == "cuda"
        self.models_frontal, self.models_lateral = paths["frontal"], paths["lateral"]
        self.models_loaded = {"f": os.path.join(folder, "frontal"), "l": os.path.join(folder, "lateral")}

    def get_model(self, view, name, cancel_check=lambda: None):
        paths = self.models_frontal if view == "frontal" else self.models_lateral
        path = paths[name]
        key = (view, path)
        if key != self._active_key:
            self.release_model()
            cancel_check()
            require_ram(max(1024 ** 3, os.path.getsize(path) * 3), "Checkpoint loading")
            model = LatencyCnnLstmModel(512, 3, 1, True, torch.device("cpu"))
            # Never download ImageNet weights: the complete state comes from this checkpoint.
            # Restricted loading avoids executing code embedded in pickle checkpoints.
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state, strict=True)
            del state, checkpoint
            model.float().eval()
            model.device = self.device
            self._active_model, self._active_key = model, key
            cancel_check()
        return self._active_model

    def load_images(self, image_f, image_l, return_normalized=False):
        shapes = [nibabel.load(path).shape for path in (image_f, image_l)]
        if any(len(shape) != 3 or min(shape) < 1 for shape in shapes):
            raise ValueError("Each NIfTI must be a nonempty 3-D H x W x T sequence.")
        length = max(shape[2] for shape in shapes)
        # Full original resolution plus padded/normalized working arrays.
        estimated = sum(shape[0] * shape[1] * length * 8 for shape in shapes) + 2 * length * 512 * 512 * 12
        require_ram(estimated, "Paired NIfTI preparation")
        return Classificator.load_images(self, image_f, image_l, return_normalized)
