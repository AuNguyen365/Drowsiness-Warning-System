import cv2
import os
import urllib.request
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

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
    Concrete FaceDetector using MediaPipe's Face Mesh API (via FaceLandmarker Tasks API).
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
        # Determine the model path
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "face_landmarker.task")
        
        # Download the model if not present
        if not os.path.exists(self.model_path):
            logger.info("Downloading face_landmarker.task model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                logger.info(f"Model downloaded successfully to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                
        # Configure FaceLandmarker
        try:
            base_options = mp_tasks.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.face_mesh = vision.FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker Tasks API initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing MediaPipe FaceLandmarker: {e}")
            self.face_mesh = None
        
    def detect_eye_landmarks(
        self, frame: np.ndarray
    ) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        if frame is None or frame.size == 0:
            logger.warning("Input frame is None or empty.")
            return None
            
        h, w = frame.shape[:2]
        
        if self.face_mesh is None:
            logger.error("FaceLandmarker is not initialized.")
            return None
            
        try:
            # Convert frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Create mediapipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            # Detect landmarks
            result = self.face_mesh.detect(mp_image)
        except Exception as e:
            logger.error(f"Error processing frame with FaceLandmarker: {e}")
            return None
            
        if not result or not result.face_landmarks:
            return None
            
        face_landmarks = result.face_landmarks[0]
        
        left_eye_points = []
        for idx in self.LEFT_EYE_INDICES:
            lm = face_landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            left_eye_points.append((x, y))
            
        right_eye_points = []
        for idx in self.RIGHT_EYE_INDICES:
            lm = face_landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            right_eye_points.append((x, y))
            
        return left_eye_points, right_eye_points
            
    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()
            logger.info("MediaPipe FaceMesh (FaceLandmarker) closed successfully.")
