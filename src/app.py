import cv2
import logging
import sys
import os
import numpy as np

# Adjust path to allow importing modules from the same directory when executed directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from camera import WebcamSource
from detector import MediaPipeFaceMeshDetector
from ear import calculate_ear, calculate_avg_ear
from alert import AudioAlertSystem
from ui import UIService
from classifier import DrowsinessClassifier

def setup_logging():
    """Configure system logging using configuration parameters."""
    log_dir = os.path.dirname(config.LOG_FILE_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    # 1. Initialize logging
    setup_logging()
    logger = logging.getLogger("WakeGuard.App")
    logger.info("Starting WakeGuard Driver Drowsiness Detection System...")
    
    # Create assets folder if it doesn't exist (for alarm.wav placement)
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # 2. Instantiate components
    camera = None
    detector = None
    alert_system = None
    
    try:
        # TODO: Instantiate WebcamSource, MediaPipeFaceMeshDetector, AudioAlertSystem, UIService, and DrowsinessClassifier
        
        # 3. State tracking variables
        consecutive_frame_counter = 0
        is_drowsy = False
        
        logger.info("Initialization complete. Entering main monitoring loop...")
        
        # TODO: Implement the real-time monitoring loop:
        # 1. Loop while camera is open
        # 2. Read frame, mirror horizontally
        # 3. Detect eye landmarks
        # 4. If landmarks are detected:
        #    a. Calculate left_ear, right_ear, and current_ear (average)
        #    b. Query the classifier to predict eye state (Open=0, Closed=1)
        #    c. If closed: increment consecutive_frame_counter
        #       - If counter >= config.CONSECUTIVE_FRAMES, set is_drowsy=True and start_alarm()
        #    d. If open: reset counter, set is_drowsy=False, and stop_alarm()
        # 5. Call ui_service.draw_hud to draw overlays (contours, EAR values, status HUD, mode indicators)
        # 6. Show the frame in a cv2 window (use config.WINDOW_TITLE)
        # 7. Listen for keypress 'Q' or 'q' to quit the application
        pass
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping system...")
    except Exception as e:
        logger.exception(f"Unhandled runtime exception: {e}")
    finally:
        # 4. Graceful Cleanup
        logger.info("Beginning system cleanup...")
        # TODO: Silently stop the alarm, release the camera, close the detector, and destroy CV2 windows
        logger.info("WakeGuard shutdown successfully.")

if __name__ == "__main__":
    main()
