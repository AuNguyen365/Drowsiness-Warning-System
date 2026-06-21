import os
import logging

# General Settings
WINDOW_TITLE = "WakeGuard - Driver Drowsiness Detection"

# Development Mock Settings
# Set to True to use dummy/stub components for detector, classifier, UI, and alarm.
USE_MOCK_COMPONENTS = True

# Detection Parameters
# EAR (Eye Aspect Ratio) threshold: Used as fallback if no ML model is found.
EAR_THRESHOLD = 0.25

# Consecutive Frames threshold: Number of consecutive frames the EAR must remain below the threshold
# (or Classified as Closed) to trigger a drowsiness alert.
CONSECUTIVE_FRAMES = 15

# Camera Settings
CAMERA_INDEX = 0

# Alert Settings
ALARM_SOUND_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "alarm.wav"
)

# Machine Learning Settings
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "drowsiness_svm.pkl")

# Logging Settings
LOG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "wakeguard.log"
)
LOG_LEVEL = logging.INFO
