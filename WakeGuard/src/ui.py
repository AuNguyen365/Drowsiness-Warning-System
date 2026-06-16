import cv2
import numpy as np
from typing import List, Tuple

class UIService:
    """
    Handles drawing HUD elements, landmark overlays, and alerts on image frames.
    Follows Single Responsibility Principle (SRP).
    """
    
    # Palette colors (BGR format)
    COLOR_SUCCESS = (0, 255, 0)      # Green
    COLOR_WARNING = (0, 165, 255)    # Orange
    COLOR_DANGER = (0, 0, 255)       # Red
    COLOR_INFO = (255, 255, 255)     # White
    COLOR_HUD_BG = (40, 40, 40)      # Dark Gray
    
    def __init__(self, ear_threshold: float):
        self.ear_threshold = ear_threshold
        # Used for creating a visual alert flashing effect
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
        # Create a copy to prevent modifying the raw frame directly if needed (or modify in-place for speed)
        out_frame = frame.copy()
        h, w, _ = out_frame.shape
        
        # 1. Draw Eye Contours and Landmarks
        if left_eye:
            self._draw_eye_contour(out_frame, left_eye, self.COLOR_SUCCESS if current_ear >= self.ear_threshold else self.COLOR_WARNING)
        if right_eye:
            self._draw_eye_contour(out_frame, right_eye, self.COLOR_SUCCESS if current_ear >= self.ear_threshold else self.COLOR_WARNING)
            
        # 2. Draw Status HUD Panel at top-left
        # Background block for readability
        cv2.rectangle(out_frame, (10, 10), (280, 110), self.COLOR_HUD_BG, -1)
        cv2.rectangle(out_frame, (10, 10), (280, 110), (100, 100, 100), 1)
        
        # Display Current EAR
        cv2.putText(
            out_frame, 
            f"EAR: {current_ear:.3f}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            self.COLOR_INFO, 
            2
        )
        
        # Display EAR Threshold
        cv2.putText(
            out_frame, 
            f"Threshold: {self.ear_threshold:.2f}", 
            (20, 65), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (180, 180, 180), 
            1
        )
        
        # Display Frame Counter status
        status_counter = f"Frames below: {consec_frame_count}/{consec_frame_max}"
        cv2.putText(
            out_frame, 
            status_counter, 
            (20, 95), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            self.COLOR_WARNING if consec_frame_count > 0 else (180, 180, 180), 
            1
        )

        # 3. Draw Drowsiness Status Banner
        self.flash_counter += 1
        if is_drowsy:
            # Flashing Red Alert banner in middle-top
            text = "DROWSINESS DETECTED!"
            font = cv2.FONT_HERSHEY_DUPLEX
            text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            
            # Flash on alternating frames
            if (self.flash_counter // 5) % 2 == 0:
                cv2.rectangle(out_frame, (text_x - 20, 15), (text_x + text_size[0] + 20, 60), self.COLOR_DANGER, -1)
                cv2.putText(
                    out_frame, 
                    text, 
                    (text_x, 48), 
                    font, 
                    1.0, 
                    self.COLOR_INFO, 
                    2
                )
        elif consec_frame_count > 0:
            # Closed warning (Yellow)
            text = "EYES CLOSED WARNING"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.rectangle(out_frame, (text_x - 15, 15), (text_x + text_size[0] + 15, 50), self.COLOR_WARNING, -1)
            cv2.putText(
                out_frame, 
                text, 
                (text_x, 42), 
                font, 
                0.8, 
                (0, 0, 0), 
                2
            )
        else:
            # Active indicator (Green)
            cv2.rectangle(out_frame, (w - 130, 15), (w - 15, 45), self.COLOR_SUCCESS, -1)
            cv2.putText(
                out_frame, 
                "ACTIVE", 
                (w - 110, 37), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 0), 
                2
            )

        # 4. Draw EAR Visual Progress Bar (Right side of screen)
        self._draw_ear_bar(out_frame, current_ear, h, w)

        # Quit instruction
        cv2.putText(
            out_frame, 
            "Press 'Q' to Exit", 
            (20, h - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (200, 200, 200), 
            1
        )
        
        return out_frame

    def _draw_eye_contour(self, frame: np.ndarray, eye_pts: List[Tuple[int, int]], color: Tuple[int, int, int]) -> None:
        """
        Draw lines linking the 6 eye coordinates and draw small keypoint circles.
        """
        pts = np.array(eye_pts, dtype=np.int32)
        # Reshape to cv2 polygon format: (number_of_vertices, 1, 2)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw closed eye outline
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
        
        # Draw individual landmarks
        for pt in eye_pts:
            cv2.circle(frame, pt, 2, color, -1)

    def _draw_ear_bar(self, frame: np.ndarray, ear: float, h: int, w: int) -> None:
        """
        Draws a graphical bar showing the current EAR value relative to the threshold.
        """
        bar_height = 150
        bar_width = 15
        x_pos = w - 30
        y_pos = 100
        
        # Draw background bar
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + bar_width, y_pos + bar_height), self.COLOR_HUD_BG, -1)
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + bar_width, y_pos + bar_height), (100, 100, 100), 1)
        
        # Map EAR (0.0 to 0.4) to bar height
        max_val = 0.40
        clipped_ear = min(max(ear, 0.0), max_val)
        fill_height = int((clipped_ear / max_val) * bar_height)
        
        # Determine fill color
        color = self.COLOR_SUCCESS
        if ear < self.ear_threshold:
            color = self.COLOR_DANGER
            
        # Draw fill level
        cv2.rectangle(
            frame, 
            (x_pos, y_pos + bar_height - fill_height), 
            (x_pos + bar_width, y_pos + bar_height), 
            color, 
            -1
        )
        
        # Draw threshold mark line
        threshold_y = y_pos + bar_height - int((self.ear_threshold / max_val) * bar_height)
        if y_pos <= threshold_y <= y_pos + bar_height:
            cv2.line(frame, (x_pos - 5, threshold_y), (x_pos + bar_width + 5, threshold_y), self.COLOR_WARNING, 2)
            cv2.putText(frame, "THR", (x_pos - 35, threshold_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLOR_WARNING, 1)
