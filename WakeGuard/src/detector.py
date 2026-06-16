import cv2
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)

class FaceDetector(ABC):
    """
    Abstract interface for face and eye landmark detection.
    Enables swapping detector backends (e.g. MediaPipe to Dlib) under the Open/Closed Principle.
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
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        logger.info("MediaPipe FaceMesh detector initialized.")
        
    def detect_eye_landmarks(
        self, frame: np.ndarray
    ) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        if frame is None:
            return None
            
        h, w, _ = frame.shape
        # MediaPipe requires RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                return None
                
            face_landmarks = results.multi_face_landmarks[0]
            
            # Map normalized coordinates to pixel coordinates
            left_eye_pts = []
            for idx in self.LEFT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                pt = (int(landmark.x * w), int(landmark.y * h))
                left_eye_pts.append(pt)
                
            right_eye_pts = []
            for idx in self.RIGHT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                pt = (int(landmark.x * w), int(landmark.y * h))
                right_eye_pts.append(pt)
                
            return left_eye_pts, right_eye_pts
            
        except Exception as e:
            logger.error(f"Error extracting face landmarks: {e}")
            return None
            
    def __del__(self):
        # Explicit clean-up of resources
        try:
            self.face_mesh.close()
            logger.debug("MediaPipe FaceMesh closed successfully.")
        except Exception:
            pass
            
    def close(self):
        self.__del__()
