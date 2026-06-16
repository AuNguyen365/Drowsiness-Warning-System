import cv2
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class FaceDetector(ABC):
    """
    Abstract interface for face and eye landmark detection.
    """
    
    @abstractmethod
    def detect_eye_landmarks(
        self, frame: np.ndarray
    ) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        """
        Processes a frame and extracts landmark coordinates for the left and right eyes.
        
        Args:
            frame (np.ndarray): Input frame in BGR format.
            
        Returns:
            Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
                A tuple containing (left_eye_points, right_eye_points) where each list contains
                6 (x, y) coordinates representing eye geometry, or None if no face is detected.
        """
        pass


class MediaPipeFaceMeshDetector(FaceDetector):
    """
    Concrete FaceDetector using MediaPipe's Face Mesh API.
    """
    
    # 6 landmarks for Left Eye: [p1, p2, p3, p4, p5, p6]
    # p1: Inner Corner, p2/p3: Upper Eyelid, p4: Outer Corner, p5/p6: Lower Eyelid
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    # 6 landmarks for Right Eye: [p1, p2, p3, p4, p5, p6]
    # p1: Outer Corner, p2/p3: Upper Eyelid, p4: Inner Corner, p5/p6: Lower Eyelid
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initialize the MediaPipe Face Mesh detector.
        """
        # TODO: Initialize MediaPipe FaceMesh solution
        self.face_mesh = None
        pass
        
    def detect_eye_landmarks(
        self, frame: np.ndarray
    ) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        # TODO: Implement eye landmark extraction
        # 1. Check if frame is None
        # 2. Convert frame from BGR to RGB (MediaPipe requirement)
        # 3. Process the frame using FaceMesh
        # 4. Extract landmarks for LEFT_EYE_INDICES and RIGHT_EYE_INDICES, scaling to pixel dimensions (w, h)
        # 5. Return (left_eye_points, right_eye_points) or None if no face is detected
        return None
            
    def close(self):
        # TODO: Safely close the FaceMesh instance
        pass
