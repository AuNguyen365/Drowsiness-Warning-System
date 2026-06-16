import cv2
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class VideoSource(ABC):
    """
    Abstract base class representing a video frame source.
    Follows Dependency Inversion Principle (DIP).
    """
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next video frame.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success status and the image frame (BGR format).
        """
        pass
        
    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check if the video source is open and active.
        
        Returns:
            bool: True if open, False otherwise.
        """
        pass
        
    @abstractmethod
    def release(self) -> None:
        """
        Release the video source resource.
        """
        pass


class WebcamSource(VideoSource):
    """
    Concrete implementation of VideoSource for capturing frames from a local webcam or file path.
    """
    
    def __init__(self, camera_index: int | str = 0):
        """
        Initialize the webcam source.
        
        Args:
            camera_index (int or str): Device index or video file path.
        """
        self.camera_index = camera_index
        self.cap = None
        self._initialize_stream()
        
    def _initialize_stream(self) -> None:
        try:
            logger.info(f"Initializing video capture stream for index/path: {self.camera_index}")
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.camera_index}")
        except Exception as e:
            logger.exception(f"Unexpected error while opening video source {self.camera_index}: {e}")
            self.cap = None
            
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Attempted to read from an uninitialized or closed camera stream.")
            return False, None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to grab frame from video source.")
                return False, None
            return True, frame
        except Exception as e:
            logger.error(f"Error reading frame from video source: {e}")
            return False, None
            
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
        
    def release(self) -> None:
        if self.cap is not None:
            logger.info(f"Releasing video capture stream: {self.camera_index}")
            self.cap.release()
            self.cap = None
