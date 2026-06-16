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
    logger.info("Starting WakeGuard Driver Drowsiness Detection System (ML Edition)...")
    
    # Create assets folder if it doesn't exist (for alarm.wav placement)
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # 2. Instantiate components
    camera = None
    detector = None
    alert_system = None
    
    try:
        camera = WebcamSource(camera_index=config.CAMERA_INDEX)
        detector = MediaPipeFaceMeshDetector()
        alert_system = AudioAlertSystem(sound_path=config.ALARM_SOUND_PATH)
        ui_service = UIService(ear_threshold=config.EAR_THRESHOLD)
        classifier = DrowsinessClassifier()
        
        if not camera.is_opened():
            logger.critical("Could not open camera stream. Exiting application.")
            return

        # 3. State tracking variables
        consecutive_frame_counter = 0
        is_drowsy = False
        
        if classifier.is_using_ml():
            logger.info("Active Detection Mode: SVM Machine Learning Classifier.")
        else:
            logger.info("Active Detection Mode: Heuristic Static Threshold Fallback.")
            
        logger.info("Initialization complete. Entering main monitoring loop...")
        
        while camera.is_opened():
            success, frame = camera.read()
            if not success or frame is None:
                logger.error("Failed to read frame from camera. Skipping cycle.")
                continue
            
            # Flip frame horizontally for natural selfie-view mirroring
            frame = cv2.flip(frame, 1)
            
            # Detect landmarks
            landmarks = detector.detect_eye_landmarks(frame)
            
            left_eye_pts = None
            right_eye_pts = None
            current_ear = 0.0
            
            if landmarks is not None:
                left_eye_pts, right_eye_pts = landmarks
                
                # Calculate individual and average EARs
                left_ear = calculate_ear(left_eye_pts)
                right_ear = calculate_ear(right_eye_pts)
                current_ear = (left_ear + right_ear) / 2.0
                
                # Classify eye closure state (0: Open, 1: Closed)
                is_closed = classifier.predict(left_ear, right_ear, current_ear)
                
                # Check for eye closure
                if is_closed == 1:
                    consecutive_frame_counter += 1
                    logger.debug(f"Eye closed detected by classifier. Counter: {consecutive_frame_counter}")
                    
                    if consecutive_frame_counter >= config.CONSECUTIVE_FRAMES:
                        if not is_drowsy:
                            logger.warning(f"Drowsiness detected! Eyes closed for {consecutive_frame_counter} frames.")
                            is_drowsy = True
                            alert_system.start_alarm()
                else:
                    if is_drowsy:
                        logger.info(f"Driver alert regained. Reseting state. EAR: {current_ear:.3f}")
                    consecutive_frame_counter = 0
                    is_drowsy = False
                    alert_system.stop_alarm()
            else:
                # Face lost / not detected
                logger.debug("Face not detected in frame.")
                
            # Draw HUD updates
            hud_frame = ui_service.draw_hud(
                frame=frame,
                left_eye=left_eye_pts,
                right_eye=right_eye_pts,
                current_ear=current_ear,
                is_drowsy=is_drowsy,
                consec_frame_count=consecutive_frame_counter,
                consec_frame_max=config.CONSECUTIVE_FRAMES
            )
            
            # Draw classification mode overlay text in the HUD
            h, w, _ = hud_frame.shape
            mode_str = "MODE: SVM (ML)" if classifier.is_using_ml() else "MODE: Static (Fallback)"
            mode_color = (0, 255, 255) if classifier.is_using_ml() else (150, 150, 150)
            cv2.putText(
                hud_frame,
                mode_str,
                (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                mode_color,
                1
            )
            
            # Additional overlay if face is not detected
            if landmarks is None:
                cv2.putText(
                    hud_frame,
                    "FACE NOT DETECTED",
                    (w // 2 - 130, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )
                
            # Display image
            cv2.imshow(config.WINDOW_TITLE, hud_frame)
            
            # Key check
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                logger.info("Quit key received. Stopping system...")
                break
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping system...")
    except Exception as e:
        logger.exception(f"Unhandled runtime exception: {e}")
    finally:
        # 4. Graceful Cleanup
        logger.info("Beginning system cleanup...")
        if alert_system is not None:
            alert_system.stop_alarm()
        if camera is not None:
            camera.release()
        if detector is not None:
            detector.close()
            
        cv2.destroyAllWindows()
        logger.info("WakeGuard shutdown successfully.")

if __name__ == "__main__":
    main()
