import os
import sys
import numpy as np
import pytest

# Add the src folder to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from detector import MediaPipeFaceMeshDetector

def test_detector_initialization():
    detector = MediaPipeFaceMeshDetector()
    assert detector.face_mesh is not None
    detector.close()

def test_detector_blank_frame():
    detector = MediaPipeFaceMeshDetector()
    # Create a blank black image (height=480, width=640, channels=3)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Run detector - should return None because there is no face
    result = detector.detect_eye_landmarks(blank_frame)
    assert result is None
    detector.close()

def test_detector_invalid_inputs():
    detector = MediaPipeFaceMeshDetector()
    
    # None input
    assert detector.detect_eye_landmarks(None) is None
    
    # Empty frame (0-sized)
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    assert detector.detect_eye_landmarks(empty_frame) is None
    
    detector.close()
