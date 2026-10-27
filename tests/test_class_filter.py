"""Model-free regression tests for the real web and CLI inference functions."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return Tensor(self.value[index])

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeModel:
    def __init__(self, label):
        self.names = {0: label}

    def __call__(self, frame, **kwargs):
        box = SimpleNamespace(xyxy=Tensor([[1, 1, 20, 20]]), conf=Tensor([0.9]), cls=Tensor([0]))
        return [SimpleNamespace(boxes=[box])]


@pytest.fixture(params=["app.py", "detect_realtime.py", "pothole-detection/app.py", "pothole-detection/detect_realtime.py"])
def inference(request, monkeypatch, tmp_path):
    source = ROOT / request.param
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(source.parent))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=object))
    monkeypatch.delitem(sys.modules, "utils", raising=False)
    spec = importlib.util.spec_from_file_location("inference_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process_frame


def test_generic_model_class_zero_is_not_a_pothole(inference):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    annotated, detections = inference(frame.copy(), FakeModel("person"), 0.5)
    assert detections == []
    assert np.array_equal(annotated, frame)


def test_explicit_pothole_class_is_detected(inference):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    _, detections = inference(frame, FakeModel("pothole"), 0.5)
    assert len(detections) == 1
    assert detections[0]["label"] == "pothole"
    assert detections[0]["conf"] == pytest.approx(0.9)
