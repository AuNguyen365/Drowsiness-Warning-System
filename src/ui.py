import cv2
import numpy as np
from typing import List, Tuple

class UIService:
    """
    Handles drawing HUD elements, landmark overlays, and alerts on image frames.
    """
    
    # Palette colors (BGR format)
    COLOR_SUCCESS = (0, 255, 0)      # Green
    COLOR_WARNING = (0, 165, 255)    # Orange
    COLOR_DANGER = (0, 0, 255)       # Red
    COLOR_INFO = (255, 255, 255)     # White
    COLOR_HUD_BG = (40, 40, 40)      # Dark Gray
    
    def __init__(self, ear_threshold: float):
        self.ear_threshold = ear_threshold
        self.flash_counter = 0

    def draw_hud(
        self, 
        frame: np.ndarray, 
        left_eye: List[Tuple[int, int]] | None, 
        right_eye: List[Tuple[int, int]] | None, 
        current_ear: float,
        is_drowsy: bool,
        consec_frame_count: int,
        consec_frame_max: int
    ) -> np.ndarray:
        """
        Draw overlays and status HUD onto the frame.
        """
        # TODO: Implement complete HUD drawing pipeline:
        # 1. Draw eye contours and landmarks if coordinates are provided (use _draw_eye_contour helper)
        # 2. Render text HUD panel (Current EAR, threshold, consecutive frames)
        # 3. Render Warning / Danger banners based on drowsiness state
        # 4. Render graphical EAR level progress bar on the right side (use _draw_ear_bar helper)
        # 5. Return the modified frame copy
        return frame

    def _draw_eye_contour(self, frame: np.ndarray, eye_pts: List[Tuple[int, int]], color: Tuple[int, int, int]) -> None:
        """
        Draw lines linking the 6 eye coordinates and draw small keypoint circles.
        """
        # TODO: Implement drawing lines using cv2.polylines (closed loop) and keypoint dots using cv2.circle
        pass

    def _draw_ear_bar(self, frame: np.ndarray, ear: float, h: int, w: int) -> None:
        """
        Draws a graphical bar showing the current EAR value relative to the threshold.
        """
        # TODO: Implement drawing a vertical progress bar on the right side of the screen
        # 1. Draw a dark background rectangle
        # 2. Draw a filled level rectangle representing current EAR (clipping values)
        # 3. Draw a line representing the warning EAR threshold
        pass
