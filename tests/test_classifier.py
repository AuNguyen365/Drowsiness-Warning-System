import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config
from classifier import DrowsinessClassifier


class StaticModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, features):
        self.features = features
        return [self.prediction]


def test_classifier_fallback_closed_when_ear_below_threshold(monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", os.path.join(os.getcwd(), "missing-model.pkl"))

    classifier = DrowsinessClassifier()

    assert classifier.is_using_ml() is False
    assert classifier.predict(0.15, 0.16, 0.155) == 1


def test_classifier_fallback_open_when_ear_above_threshold(monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", os.path.join(os.getcwd(), "missing-model.pkl"))

    classifier = DrowsinessClassifier()

    assert classifier.is_using_ml() is False
    assert classifier.predict(0.31, 0.33, 0.32) == 0


def test_classifier_uses_loaded_model_prediction(monkeypatch):
    monkeypatch.setattr(DrowsinessClassifier, "_load_model", lambda self: setattr(self, "model", StaticModel(1)))

    classifier = DrowsinessClassifier()

    assert classifier.is_using_ml() is True
    assert classifier.predict(0.31, 0.33, 0.32) == 1
    assert classifier.model.features == [[0.31, 0.33, 0.32]]
