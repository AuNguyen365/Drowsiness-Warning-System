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
    h, w, _ = frame.shape
    # Draw background panel at the top
    cv2.rectangle(frame, (0, 0), (w, 80), bg_color, -1)
    
    cv2.putText(
        frame,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    if text2:
        cv2.putText(
            frame,
            text2,
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1
        )

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
    
    camera = WebcamSource(camera_index=config.CAMERA_INDEX)
    detector = MediaPipeFaceMeshDetector()
    
    if not camera.is_opened():
        print("Loi: Khong the mo camera. Vui long kiem tra lai thiet bi.")
        return
        
    dataset_records = []
    
    try:
        # State Machine for data collection:
        # 0: Preparation / Get ready
        # 1: Collect Eyes Open (10s)
        # 2: Transition / Prepare for closed
        # 3: Collect Eyes Closed (10s)
        # 4: Save & Finished
        
        state = 0
        state_start_time = time.time()
        collection_duration = 10.0 # 10 seconds for each state
        prep_duration = 5.0 # 5 seconds to get ready
        
        print("Camera dang mo. Hay can chinh khuon mat cua ban vao giua khung hinh...")
        
        while camera.is_opened():
            success, frame = camera.read()
            if not success or frame is None:
                continue
                
            frame = cv2.flip(frame, 1)
            landmarks = detector.detect_eye_landmarks(frame)
            
            elapsed = time.time() - state_start_time
            left_eye_pts = None
            right_eye_pts = None
            current_ear = 0.0
            
            if landmarks is not None:
                left_eye_pts, right_eye_pts = landmarks
                current_ear = calculate_avg_ear(left_eye_pts, right_eye_pts)
                
                # Draw eye outlines
                pts_l = np.array(left_eye_pts, dtype=np.int32).reshape((-1, 1, 2))
                pts_r = np.array(right_eye_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts_l], True, (0, 255, 0), 1)
                cv2.polylines(frame, [pts_r], True, (0, 255, 0), 1)
                
            # State management
            if state == 0:
                # Get Ready
                remaining = int(prep_duration - elapsed)
                draw_instruction(frame, f"CHUAN BI THU THAP EYES OPEN - CON {remaining}S", "Hay nhin thang vao camera va giu mat mo tu nhien.")
                if remaining <= 0:
                    state = 1
                    state_start_time = time.time()
                    print("Bat dau thu thap du lieu mat MO...")
                    
            elif state == 1:
                # Collect Open
                remaining = int(collection_duration - elapsed)
                draw_instruction(frame, f"DANG THU THAP EYES OPEN - CON {remaining}S", f"Giu mat mo. EAR hien tai: {current_ear:.3f}", (0, 120, 0))
                
                if landmarks is not None and current_ear > 0:
                    left_ear = calculate_ear(left_eye_pts)
                    right_ear = calculate_ear(right_eye_pts)
                    dataset_records.append([left_ear, right_ear, current_ear, 0])
                    
                if remaining <= 0:
                    state = 2
                    state_start_time = time.time()
                    print("Bat dau chuan bi nham mat...")
                    
            elif state == 2:
                # Prep for Closed
                remaining = int(prep_duration - elapsed)
                draw_instruction(frame, f"CHUAN BI THU THAP EYES CLOSED - CON {remaining}S", "Chuan bi nham mat lai sau khi ket thuc dem nguoc.")
                if remaining <= 0:
                    state = 3
                    state_start_time = time.time()
                    print("Bat dau thu thap du lieu mat NHAM...")
                    
            elif state == 3:
                # Collect Closed
                remaining = int(collection_duration - elapsed)
                draw_instruction(frame, f"DANG THU THAP EYES CLOSED - CON {remaining}S", f"Nham mat tu nhien. EAR hien tai: {current_ear:.3f}", (0, 0, 150))
                
                if landmarks is not None and current_ear > 0:
                    left_ear = calculate_ear(left_eye_pts)
                    right_ear = calculate_ear(right_eye_pts)
                    dataset_records.append([left_ear, right_ear, current_ear, 1])
                    
                if remaining <= 0:
                    state = 4
                    break
                    
            cv2.imshow("WakeGuard - Thu Thap Du Lieu Huan Luyen", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Da huy qua trinh thu thap du lieu.")
                break
                
    except Exception as e:
        logger.exception(f"Loi xay ra trong qua trinh thu thap: {e}")
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        
    # Import numpy here locally to avoid errors before pip finishes installation
    import numpy as np
    
    if state == 4 and len(dataset_records) > 0:
        print(f"Thu thap thanh cong {len(dataset_records)} ban ghi.")
        
        # Ask to append or overwrite
        mode = 'w'
        write_header = True
        if os.path.exists(config.DATASET_PATH):
            print("Tep dataset.csv da ton tai. Bạn muon:")
            print("  [1] Ghi de tep cu (Overwrite)")
            print("  [2] Ghi noi tiep vao tep cu (Append)")
            # Standard auto-select 2 for safety, or check if user input is possible.
            # Since running in an agent terminal we'll write a prompt that falls back to append
            print("-> Mac dinh: Ghi noi tiep (Append)")
            mode = 'a'
            write_header = False
            
        try:
            with open(config.DATASET_PATH, mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["left_ear", "right_ear", "avg_ear", "label"])
                writer.writerows(dataset_records)
            print(f"Da luu du lieu thanh cong vao: {config.DATASET_PATH}")
        except Exception as e:
            print(f"Loi ghi tep CSV: {e}")
    else:
        print("Khong thu thap duoc du lieu.")

if __name__ == "__main__":
    import numpy as np
    main()

