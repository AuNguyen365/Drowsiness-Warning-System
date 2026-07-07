# pyrefly: ignore [missing-import]
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
        logger.info("Initializing actual system components...")
        
        # Instantiate actual components directly
        camera = WebcamSource(config.CAMERA_INDEX)
        if not camera.is_opened():
            raise RuntimeError("WebcamSource index could not be opened.")
        logger.info("WebcamSource initialized successfully.")
        
        detector = MediaPipeFaceMeshDetector()
        # Run dry run with dummy empty frame to verify initialization
        dry_run_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector.detect_eye_landmarks(dry_run_frame)
        logger.info("MediaPipeFaceMeshDetector initialized successfully.")
        
        classifier = DrowsinessClassifier()
        logger.info(f"DrowsinessClassifier initialized. Using ML: {classifier.is_using_ml()}")
        
        ui_service = UIService(config.EAR_THRESHOLD)
        logger.info("UIService initialized successfully.")
        
        alert_system = AudioAlertSystem(config.ALARM_SOUND_PATH)
        logger.info("AudioAlertSystem initialized successfully.")
            
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
