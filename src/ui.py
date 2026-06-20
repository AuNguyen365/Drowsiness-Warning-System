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
        self.flash_frame_count = 0
        self.last_state = None  # Track drowsy/warning state changes

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
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw eye contours
        if left_eye:
            self._draw_eye_contour(out, left_eye, self.COLOR_SUCCESS if not is_drowsy else self.COLOR_WARNING)
        if right_eye:
            self._draw_eye_contour(out, right_eye, self.COLOR_SUCCESS if not is_drowsy else self.COLOR_WARNING)

        # HUD panel (top-left)
        panel_w, panel_h = 260, 80
        pad = 10
        panel_x, panel_y = pad, pad
        cv2.rectangle(out, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), self.COLOR_HUD_BG, -1)
        cv2.putText(out, f"EAR: {current_ear:.3f}", (panel_x + 12, panel_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_INFO, 1, cv2.LINE_AA)
        cv2.putText(out, f"Thresh: {self.ear_threshold:.3f}", (panel_x + 12, panel_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_INFO, 1, cv2.LINE_AA)
        cv2.putText(out, f"Frames: {consec_frame_count}/{consec_frame_max}", (panel_x + 12, panel_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_INFO, 1, cv2.LINE_AA)

        # EAR bar on the right
        self._draw_ear_bar(out, current_ear, h, w)

        # Warning / Danger banner (with frame-based flashing effect)
        current_state = "drowsy" if is_drowsy else ("warning" if current_ear < self.ear_threshold else "normal")
        
        # Reset flash counter when state changes
        if current_state != self.last_state:
            self.flash_frame_count = 0
            self.last_state = current_state
        
        # Show banner with frame-based cycling: 10 frames on / 10 frames off
        if current_state != "normal":
            cycle_pos = self.flash_frame_count % 20
            
            if cycle_pos < 10:  # First 10 frames: show banner
                if is_drowsy:
                    text = "DROWSY"
                    color = self.COLOR_DANGER
                    cv2.putText(out, text, (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_DUPLEX, 2.2, color, 4, cv2.LINE_AA)
                else:
                    text = "WARNING"
                    color = self.COLOR_WARNING
                    cv2.putText(out, text, (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3, cv2.LINE_AA)
            
            self.flash_frame_count += 1

        return out

    def _draw_eye_contour(self, frame: np.ndarray, eye_pts: List[Tuple[int, int]], color: Tuple[int, int, int]) -> None:
        """
        Draw lines linking the 6 eye coordinates and draw small keypoint circles.
        """
        if not eye_pts:
            return
        pts = np.array(eye_pts, dtype=np.int32)
        if pts.shape[0] >= 3:
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
        for (x, y) in eye_pts:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    def _draw_ear_bar(self, frame: np.ndarray, ear: float, h: int, w: int) -> None:
        """
        Draws a graphical bar showing the current EAR value relative to the threshold.
        """
        # Bar dimensions
        bar_w = 28
        bar_h = int(h * 0.5)
        x0 = w - bar_w - 12
        y0 = int(h * 0.1)
        # Background
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (30, 30, 30), -1)

        # Map EAR (commonly ~0.1-0.4) to bar level
        min_ear, max_ear = 0.08, 0.4
        clipped = max(min(ear, max_ear), min_ear)
        rel = (clipped - min_ear) / (max_ear - min_ear)
        level_h = int(rel * bar_h)
        # Filled portion (from bottom)
        cv2.rectangle(frame, (x0, y0 + bar_h - level_h), (x0 + bar_w, y0 + bar_h), self.COLOR_SUCCESS, -1)

        # Threshold line
        thr = self.ear_threshold
        thr_rel = (max(min(thr, max_ear), min_ear) - min_ear) / (max_ear - min_ear)
        thr_y = int(y0 + bar_h - thr_rel * bar_h)
        cv2.line(frame, (x0, thr_y), (x0 + bar_w, thr_y), self.COLOR_WARNING, 2)

        # Label
        cv2.putText(frame, f"EAR", (x0 - 34, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_INFO, 1, cv2.LINE_AA)
