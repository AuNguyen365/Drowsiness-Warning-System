import cv2
import logging
import sys
import os
import numpy as np
import warnings

# Suppress sklearn UserWarning regarding feature names during inference
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
from typing import List, Tuple, Optional

# Adjust path to allow importing modules from the same directory when executed directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from camera import WebcamSource
from detector import MediaPipeFaceMeshDetector
from ear import calculate_ear, calculate_avg_ear
from alert import AudioAlertSystem
from ui import UIService
from classifier import DrowsinessClassifier

logger = logging.getLogger(__name__)

# --- STUB/MOCK CLASSES FOR INDEPENDENT DEVELOPMENT (DAY 2) ---

class DummyCameraSource:
    def __init__(self):
        self.frame_count = 0
        logger.info("[MOCK CAMERA] Initializing dummy camera source...")
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        self.frame_count += 1
        if self.frame_count > 300:  # Limit execution to 300 frames (approx 10s)
            logger.info("[MOCK CAMERA] Reached mock frame limit. Stopping stream.")
            return False, None
        # Create a black frame of size 480x640
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw some grid/content so the screen isn't empty
        cv2.putText(frame, f"Mock Frame #{self.frame_count}", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        # Simple animation: draw a moving circle
        circle_x = int(320 + 100 * np.sin(self.frame_count * 0.1))
        cv2.circle(frame, (circle_x, 240), 20, (0, 0, 255), -1)
        return True, frame
        
    def is_opened(self) -> bool:
        return self.frame_count < 300
        
    def release(self) -> None:
        logger.info("[MOCK CAMERA] Releasing mock camera source.")


class DummyDetector:
    def detect_eye_landmarks(
        self, frame: np.ndarray
    ) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        # Return mock eye points (6 coordinates for left and right eyes)
        # Coordinates are chosen to be inside typical frame sizes (640x480)
        left_eye = [(200, 240), (215, 230), (230, 230), (245, 240), (230, 250), (215, 250)]
        right_eye = [(350, 240), (365, 230), (380, 230), (395, 240), (380, 250), (365, 250)]
        return left_eye, right_eye
        
    def close(self):
        logger.info("[MOCK DETECTOR] Closing mock detector resources.")


class DummyClassifier:
    def __init__(self):
        self.frame_count = 0
        
    def predict(self, left_ear: float, right_ear: float, avg_ear: float) -> int:
        self.frame_count += 1
        # Simulate drowsiness: Awake for 90 frames, closed for 30 frames (periodic simulation)
        if (self.frame_count % 120) >= 90:
            return 1  # Closed
        return 0  # Open
        
    def is_using_ml(self) -> bool:
        return False


class DummyUI:
    def __init__(self, ear_threshold: float):
        self.ear_threshold = ear_threshold
        
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
        h, w = frame.shape[:2]
        
        # Draw translucent HUD background at the top
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, 150), (40, 40, 40), -1)
        cv2.addWeighted(hud_overlay, 0.4, frame, 0.6, 0, frame)
        
        # Display main system info
        cv2.putText(frame, "WakeGuard HUD [MOCK ACTIVE]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"EAR: {current_ear:.3f} (Threshold: {self.ear_threshold})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        status_text = "DROWSY / BUON NGU!" if is_drowsy else "AWAKE / TINH TAO"
        color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        cv2.putText(frame, f"STATUS: {status_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Frames: {consec_frame_count}/{consec_frame_max}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw mock eyes
        if left_eye and right_eye:
            for pt in left_eye:
                cv2.circle(frame, pt, 3, (0, 255, 0), -1)
            for pt in right_eye:
                cv2.circle(frame, pt, 3, (0, 255, 0), -1)
                
            # Draw lines between points to simulate contour
            cv2.polylines(frame, [np.array(left_eye)], True, (255, 255, 0), 1)
            cv2.polylines(frame, [np.array(right_eye)], True, (255, 255, 0), 1)
            
        # Draw simple EAR level progress bar on the right side
        bar_x = w - 40
        bar_y_start = 180
        bar_y_end = h - 50
        bar_height = bar_y_end - bar_y_start
        cv2.rectangle(frame, (bar_x, bar_y_start), (bar_x + 15, bar_y_end), (100, 100, 100), -1)
        
        # Current EAR level height
        ear_level = int(min(max(current_ear, 0.0), 0.5) / 0.5 * bar_height)
        cv2.rectangle(frame, (bar_x, bar_y_end - ear_level), (bar_x + 15, bar_y_end), (0, 255, 0), -1)
        
        # Draw threshold marker line
        thresh_y = int(self.ear_threshold / 0.5 * bar_height)
        cv2.line(frame, (bar_x - 5, bar_y_end - thresh_y), (bar_x + 20, bar_y_end - thresh_y), (0, 0, 255), 2)
        
        return frame


class DummyAlertSystem:
    def __init__(self, sound_path: str):
        self.sound_path = sound_path
        self.is_active = False
        
    def start_alarm(self) -> None:
        if not self.is_active:
            self.is_active = True
            logger.info(f"[MOCK ALARM] Starting alarm sound. Target File: {self.sound_path}")
            
    def stop_alarm(self) -> None:
        if self.is_active:
            self.is_active = False
            logger.info("[MOCK ALARM] Stopping alarm sound.")


# --- END OF MOCK CLASSES ---

def setup_logging():
    """Configure system logging using configuration parameters."""
    log_dir = os.path.dirname(config.LOG_FILE_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE_PATH, mode='w', encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    # 1. Initialize logging
    setup_logging()
    logger.info("Starting WakeGuard Driver Drowsiness Detection System...")
    
    # Create assets folder if it doesn't exist (for alarm.wav placement)
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # 2. Instantiate components
    camera = None
    detector = None
    classifier = None
    ui_service = None
    alert_system = None
    
    try:
        # Check configuration for mock components
        if config.USE_MOCK_COMPONENTS:
            logger.info("Using development MOCK/STUB components as configured.")
            camera = DummyCameraSource()
            detector = DummyDetector()
            classifier = DummyClassifier()
            ui_service = DummyUI(config.EAR_THRESHOLD)
            alert_system = DummyAlertSystem(config.ALARM_SOUND_PATH)
        else:
            logger.info("Initializing actual system components with dynamic fallbacks...")
            
            # Try camera
            try:
                camera = WebcamSource(config.CAMERA_INDEX)
                if not camera.is_opened():
                    raise RuntimeError("WebcamSource index could not be opened.")
                logger.info("Actual WebcamSource initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize actual camera: {e}. Falling back to DummyCameraSource.")
                camera = DummyCameraSource()
                
            # Try detector
            try:
                detector = MediaPipeFaceMeshDetector()
                # Run dry run with dummy empty frame to verify initialization
                dry_run_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                detector.detect_eye_landmarks(dry_run_frame)
                logger.info("Actual MediaPipeFaceMeshDetector initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize actual detector: {e}. Falling back to DummyDetector.")
                detector = DummyDetector()
                
            # Try classifier
            try:
                classifier = DrowsinessClassifier()
                logger.info(f"Actual DrowsinessClassifier initialized. Using ML: {classifier.is_using_ml()}")
            except Exception as e:
                logger.error(f"Failed to initialize actual classifier: {e}. Falling back to DummyClassifier.")
                classifier = DummyClassifier()
                
            # Try UIService
            try:
                ui_service = UIService(config.EAR_THRESHOLD)
                logger.info("Actual UIService initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize actual UIService: {e}. Falling back to DummyUI.")
                ui_service = DummyUI(config.EAR_THRESHOLD)
                
            # Try AudioAlertSystem
            try:
                alert_system = AudioAlertSystem(config.ALARM_SOUND_PATH)
                logger.info("Actual AudioAlertSystem initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize actual AudioAlertSystem: {e}. Falling back to DummyAlertSystem.")
                alert_system = DummyAlertSystem(config.ALARM_SOUND_PATH)
            
        # 3. State tracking variables
        consecutive_frame_counter = 0
        is_drowsy = False
        
        logger.info("Initialization complete. Entering main monitoring loop...")
        
        while camera.is_opened():
            success, frame = camera.read()
            if not success or frame is None:
                # Wait briefly to prevent spinning if reading fails intermittently
                cv2.waitKey(10)
                continue
                
            # Mirror horizontally for natural viewer reflection
            frame = cv2.flip(frame, 1)
            
            # Detect eye landmarks
            landmarks = detector.detect_eye_landmarks(frame)
            
            left_eye, right_eye = None, None
            current_ear = 0.0
            
            if landmarks is not None:
                left_eye, right_eye = landmarks
                # Calculate EAR for both eyes and average
                current_ear = calculate_avg_ear(left_eye, right_eye)
                
                # Calculate individual EARs for classifier query
                left_ear_val = calculate_ear(left_eye)
                right_ear_val = calculate_ear(right_eye)
                
                # Query classifier to predict eye state (Open=0, Closed=1)
                prediction = classifier.predict(left_ear_val, right_ear_val, current_ear)
                
                if prediction == 1:
                    consecutive_frame_counter += 1
                    if consecutive_frame_counter >= config.CONSECUTIVE_FRAMES:
                        if not is_drowsy:
                            is_drowsy = True
                            logger.warning(f"Drowsiness detected! (Closed eye frames: {consecutive_frame_counter})")
                            alert_system.start_alarm()
                else:
                    consecutive_frame_counter = 0
                    if is_drowsy:
                        is_drowsy = False
                        logger.info("Driver is awake. Stopping alert.")
                        alert_system.stop_alarm()
            else:
                # No face/eyes detected
                consecutive_frame_counter = 0
                if is_drowsy:
                    is_drowsy = False
                    logger.info("Driver face lost. Stopping alert.")
                    alert_system.stop_alarm()
                    
            # Draw UI/HUD overlays
            frame = ui_service.draw_hud(
                frame, 
                left_eye, 
                right_eye, 
                current_ear, 
                is_drowsy, 
                consecutive_frame_counter, 
                config.CONSECUTIVE_FRAMES
            )
            
            # Display output frame
            cv2.imshow(config.WINDOW_TITLE, frame)
            
            # Press Q or q key to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                logger.info("Quit key 'q'/'Q' pressed. Exiting monitoring loop.")
                break
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping system...")
    except Exception as e:
        logger.exception(f"Unhandled runtime exception: {e}")
    finally:
        # 4. Graceful Cleanup
        logger.info("Beginning system cleanup...")
        if alert_system is not None:
            try:
                alert_system.stop_alarm()
            except Exception as ex:
                logger.error(f"Error stopping alarm during cleanup: {ex}")
                
        if camera is not None:
            try:
                camera.release()
            except Exception as ex:
                logger.error(f"Error releasing camera during cleanup: {ex}")
                
        if detector is not None and hasattr(detector, 'close'):
            try:
                detector.close()
            except Exception as ex:
                logger.error(f"Error closing detector during cleanup: {ex}")
                
        cv2.destroyAllWindows()
        logger.info("WakeGuard shutdown successfully.")

if __name__ == "__main__":
    main()
