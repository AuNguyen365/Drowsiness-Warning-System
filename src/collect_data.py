import cv2
import time
import csv
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from camera import WebcamSource
from detector import MediaPipeFaceMeshDetector
from ear import calculate_avg_ear, calculate_ear

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("WakeGuard.DataCollection")

def draw_instruction(frame, text, text2=None, bg_color=(0, 0, 0)):
    """Draw instructions overlay text on frame."""
    # TODO: Implement optional instruction rendering helper
    pass

def main():
    print("=========================================================")
    print("       WAKEGUARD INTERACTIVE DATA COLLECTION TOOL       ")
    print("=========================================================")
    print("Cong cu nay se thu thap chi so EAR cua ban:")
    print("  1. Mo mat binh thuong trong 10 giay (Nhan 0).")
    print("  2. Nham mat tu nhien trong 10 giay (Nhan 1).")
    print("Du lieu se duoc ghi vao thu muc: " + config.DATASET_PATH)
    print("=========================================================")
    
    # Ensure data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # TODO: Instantiate WebcamSource and MediaPipeFaceMeshDetector
    camera = None
    detector = None
    
    dataset_records = []
    
    try:
        # State Machine for data collection:
        # 0: Preparation / Get ready (5s)
        # 1: Collect Eyes Open (10s)
        # 2: Transition / Prepare for closed (5s)
        # 3: Collect Eyes Closed (10s)
        # 4: Save & Finished
        
        state = 0
        state_start_time = time.time()
        collection_duration = 10.0
        prep_duration = 5.0
        
        # TODO: Implement the capture state machine loop:
        # 1. Loop while camera is open
        # 2. Read frame, mirror horizontally
        # 3. Detect eye landmarks
        # 4. Depending on state (0 to 3), calculate EAR, overlay guidelines on the frame,
        #    and record eye feature samples [left_ear, right_ear, avg_ear, label]
        #    into dataset_records list.
        # 5. Break loop when state 4 is reached.
        pass
                
    except Exception as e:
        logger.exception(f"Loi xay ra trong qua trinh thu thap: {e}")
    finally:
        # TODO: Clean up camera, detector and windows
        pass
        
    # TODO: Write collected records to data/dataset.csv
    # 1. Check if dataset_records is not empty
    # 2. Open CSV file in append/write mode
    # 3. Write CSV rows containing columns: ["left_ear", "right_ear", "avg_ear", "label"]
    pass

if __name__ == "__main__":
    main()
