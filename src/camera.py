import cv2
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class VideoSource(ABC):
    """
    Abstract base class representing a video frame source.
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
            logger.info(f"Initializing video capture stream with index/path: {self.camera_index}...")
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.camera_index}")
                self.cap = None
            else:
                logger.info("Video capture stream initialized successfully.")
        except Exception as e:
            logger.exception(f"Error during video capture initialization: {e}")
            self.cap = None
            
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened():
            logger.warning("Capture stream is not open. Attempting to re-initialize...")
            self._initialize_stream()
            if not self.is_opened():
                logger.error("Re-initialization failed. Cannot read frame.")
                return False, None
                
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Failed to retrieve frame from video source.")
                return False, None
            return True, frame
        except Exception as e:
            logger.exception(f"Error reading frame from video source: {e}")
            return False, None
            
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
        
    def release(self) -> None:
        if self.is_opened():
            logger.info("Releasing video capture stream...")
            try:
                self.cap.release()
                logger.info("Video capture stream released successfully.")
            except Exception as e:
                logger.exception(f"Error releasing video capture stream: {e}")
        self.cap = None
