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
        # TODO: Initialize the camera capture stream
        pass
        
    def _initialize_stream(self) -> None:
        # TODO: Implement stream opening using cv2.VideoCapture with error handling
        pass
            
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # TODO: Implement frame reading. Return (success_status, frame)
        # Remember to handle exceptions and check if the capture stream is open
        return False, None
            
    def is_opened(self) -> bool:
        # TODO: Return True if video capture stream is open, False otherwise
        return False
        
    def release(self) -> None:
        # TODO: Release the cv2.VideoCapture stream and perform cleanup
        pass
