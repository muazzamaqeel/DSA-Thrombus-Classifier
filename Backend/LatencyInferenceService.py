import hashlib
import json
import os
import threading
import time
import uuid

import torch

from LatencyDiagnostics import BACKEND_REVISION, get_latency_info
from LatencyTimer import run_timed_model
from LatencyRuntimeLimits import CNN_GROUPS_PER_BATCH, GIB, CPU_THREADS
from LatencyRuntime import LatencyRuntime


class LatencyCancelled(RuntimeError):
    pass


def file_sha256(path, cancel_check=lambda: None):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            cancel_check()
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    # memoryview avoids another full-volume bytes allocation.
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


class LatencyInferenceService:
    def __init__(self, classificator):
        self.classificator = LatencyRuntime()
        self._lock = self.classificator.execution_lock
        self._saved_settings = None
        self._prepared = {}
        self._run_id = None
        self._cancel = threading.Event()
        self._warmed = set()
        self._environment_json = ''
        self._device_label = 'cpu'

    def _save_settings(self):
        targets = [(torch.backends.cudnn, 'benchmark')]
        if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
            targets += [(torch.backends, 'fp32_precision'),
                        (torch.backends.cuda.matmul, 'fp32_precision'),
                        (torch.backends.cudnn, 'fp32_precision')]
        else:
            targets += [(torch.backends.cuda.matmul, 'allow_tf32'),
                        (torch.backends.cudnn, 'allow_tf32')]
        self._saved_settings = (torch.get_num_threads(),
            torch.cuda.current_device() if torch.cuda.is_available() else None,
            [(obj, name, getattr(obj, name)) for obj, name in targets])

    def _restore_settings(self):
        if self._saved_settings is None:
            return
        threads, device, values = self._saved_settings
        for obj, name, value in values:
            setattr(obj, name, value)
        torch.set_num_threads(threads)
        if device is not None:
            torch.cuda.set_device(device)
        self._saved_settings = None

    def cancel(self, run_id):
        # Intentionally no execution lock: cancellation must reach an active forward pass.
        if run_id == self._run_id:
            self._cancel.set()

    def _check_cancel(self):
        if self._cancel.is_set():
            raise LatencyCancelled('Test stopped. The active GPU operation has finished safely.')

    def _require_run(self, run_id):
        if not self._run_id or run_id != self._run_id:
            raise ValueError('Latency run expired. Start a new test.')
        self._check_cancel()

    def configure_execution(self, mode, model_folder, device_index=0, run_id=None):
        with self._lock:
            if self._run_id is not None:
                raise ValueError("A latency run is already active. Stop/end it first.")
            self._run_id = run_id or uuid.uuid4().hex
            self._save_settings()
            self._cancel.clear()
            self._prepared.clear()
            self._warmed.clear()
            self.classificator.preparedImages.clear()
            mode = str(mode).upper()
            if mode not in ('CPU', 'GPU'):
                raise ValueError('Choose CPU or GPU.')
            if mode == 'GPU':
                if not torch.cuda.is_available():
                    raise ValueError('CUDA is unavailable in this backend. No CPU fallback was used.')
                if type(device_index) is not int or not 0 <= device_index < torch.cuda.device_count():
                    raise ValueError('Selected GPU is unavailable.')
            device = torch.device(f'cuda:{device_index}' if mode == 'GPU' else 'cpu')
            self.classificator.load_models(model_folder, device_override=str(device))
            expected = {f'fold{i}.pt' for i in range(1, 6)}
            if set(self.classificator.models_frontal) != expected:
                raise ValueError('Latency requires exactly fold1.pt through fold5.pt in both views.')
            # Disable autotuning: changing temporal sizes must not trigger costly searches.
            torch.set_num_threads(CPU_THREADS)
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
                torch.backends.fp32_precision = 'ieee'
                torch.backends.cuda.matmul.fp32_precision = 'ieee'
                torch.backends.cudnn.fp32_precision = 'ieee'
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            self._device_label = f'{device} | {torch.cuda.get_device_name(device)}' if mode == 'GPU' else 'cpu'
            if mode == 'GPU':
                # Prove actual kernel execution; availability alone is insufficient.
                with torch.cuda.device(device):
                    probe = torch.ones((16, 16), device=device)
                    probe = probe @ probe
                    torch.cuda.synchronize(device)
                    del probe
            info = get_latency_info()
            info.update({'Precision': 'FP32 (TF32 disabled)', 'SelectedDevice': str(device),
                         'CudnnBenchmark': False, 'CnnGroupsPerBatch': CNN_GROUPS_PER_BATCH,
                         'WarmupPolicy': '1 per model/view on first case only', 'TimedRuns': 1,
                         'TimingScope': 'Sum of resident-input view classification wall times plus client soft-vote; '
                                        'excludes disk loading, preprocessing, transfers, warm-up and HTTP.',
                         'ModelSha256': {f'{view}/{name}': file_sha256(path, self._check_cancel)
                            for view, paths in [('frontal', self.classificator.models_frontal),
                                                ('lateral', self.classificator.models_lateral)]
                            for name, path in paths.items()}})
            self._check_cancel()
            self._environment_json = json.dumps(info, sort_keys=True)
            return self._metadata()

    def _metadata(self):
        gpu = self.classificator.device.type == 'cuda'
        return {'BackendRevision': BACKEND_REVISION, 'RunId': self._run_id,
                'ExecutionProvider': 'GPU' if gpu else 'CPU', 'TimingDevice': self._device_label,
                'TimingMethod': 'Sum of single synchronized view classifications + soft-vote',
                'Precision': 'FP32 (TF32 disabled)', 'WarmupRuns': 1, 'TimedRuns': 1,
                'ModelResidency': 'one checkpoint at a time', 'EnvironmentJson': self._environment_json}

    def prepare_images(self, frontal_path, lateral_path, run_id):
        with self._lock:
            self._require_run(run_id)
            self._prepared.clear()
            self.classificator.release_model()
            if not all(isinstance(p, str) and os.path.isfile(p) for p in (frontal_path, lateral_path)):
                raise ValueError('Both input files must exist on the backend computer.')
            start = time.perf_counter()
            prepared, preview = self.classificator.load_images(frontal_path, lateral_path, False)
            del preview
            self._check_cancel()
            frontal = prepared['image'].unsqueeze(0).float().contiguous()
            lateral = prepared['imageOtherView'].unsqueeze(0).float().contiguous()
            del prepared
            preprocess_ms = (time.perf_counter() - start) * 1000.0
            if not torch.isfinite(frontal).all() or not torch.isfinite(lateral).all():
                raise ValueError('Non-finite normalized input (for example a constant sequence). Case rejected.')
            details = {'PreprocessMilliseconds': preprocess_ms,
                       'FrontalShape': list(frontal.shape), 'LateralShape': list(lateral.shape),
                       'FrontalFileSha256': file_sha256(frontal_path, self._check_cancel),
                       'LateralFileSha256': file_sha256(lateral_path, self._check_cancel),
                       'FrontalTensorSha256': tensor_sha256(frontal),
                       'LateralTensorSha256': tensor_sha256(lateral)}
            self._check_cancel()
            self._prepared[(frontal_path, lateral_path)] = (frontal, lateral)
            return {'PreparationJson': json.dumps(details, sort_keys=True)}

    def release_images(self, frontal_path, lateral_path, run_id):
        with self._lock:
            if run_id == self._run_id:
                self._prepared.clear()
                self.classificator.release_model()

    def end_run(self, run_id):
        with self._lock:
            if run_id == self._run_id:
                self._prepared.clear()
                self.classificator.release_model()
                self._restore_settings()
                self._run_id = None

    def _run_view(self, view, model_name, cpu_image):
        self._check_cancel()
        load_start = time.perf_counter()
        model = self.classificator.get_model(view, model_name, self._check_cancel)
        load_ms = (time.perf_counter() - load_start) * 1000.0
        device = self.classificator.device
        image = None
        try:
            if device.type == 'cuda':
                free, total = torch.cuda.mem_get_info(device)
                weights = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
                # A conservative reserve for CNN activations/workspace and the desktop.
                reserve = max(GIB, int(total * 0.25))
                if free < weights + cpu_image.numel() * 4 + reserve:
                    raise MemoryError('Insufficient free VRAM for one model, input and reserve. Test stopped; no CPU fallback.')
                torch.cuda.reset_peak_memory_stats(device)
            transfer_start = time.perf_counter()
            model.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            model_ms = (time.perf_counter() - transfer_start) * 1000.0
            transfer_start = time.perf_counter()
            image = cpu_image.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            input_ms = (time.perf_counter() - transfer_start) * 1000.0
            key = (view, model_name)
            result = run_timed_model(model, image, device, warmup_runs=0 if key in self._warmed else 1,
                                     cancel_check=self._check_cancel)
            self._warmed.add(key)
            result.update({'CheckpointLoadMilliseconds': load_ms, 'ModelTransferMilliseconds': model_ms,
                           'InputTransferMilliseconds': input_ms, 'InputShape': list(image.shape),
                           'CnnGroupsPerBatch': CNN_GROUPS_PER_BATCH,
                           'PeakAllocatedVramBytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                           'PeakReservedVramBytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0})
            return result
        finally:
            del image
            # Release, rather than making a second CPU copy of the GPU weights.
            del model
            self.classificator.release_model()

    def classify(self, model_name, frontal_path, lateral_path, run_id):
        with self._lock:
            self._require_run(run_id)
            if model_name not in self.classificator.models_frontal:
                raise ValueError('Unknown model checkpoint.')
            images = self._prepared.get((frontal_path, lateral_path))
            if images is None:
                raise ValueError('Prepare images before classification.')
            start = time.perf_counter()
            frontal = self._run_view('frontal', model_name, images[0])
            lateral = self._run_view('lateral', model_name, images[1])
            result = self._metadata()
            result.update({'OutputFrontal': [frontal['ModelOutput']], 'OutputLateral': [lateral['ModelOutput']],
                           'FrontalInferenceMilliseconds': frontal['ClassificationMilliseconds'],
                           'LateralInferenceMilliseconds': lateral['ClassificationMilliseconds'],
                           'FrontalForwardMilliseconds': frontal['ForwardMilliseconds'],
                           'LateralForwardMilliseconds': lateral['ForwardMilliseconds'],
                           'FrontalBenchmarkJson': json.dumps(frontal, sort_keys=True),
                           'LateralBenchmarkJson': json.dumps(lateral, sort_keys=True),
                           'BackendRequestMilliseconds': (time.perf_counter() - start) * 1000.0})
            return result
