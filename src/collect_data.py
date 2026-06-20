import cv2
import time
import csv
import os
import sys
import logging
import argparse
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("WakeGuard.DataCollection")

CSV_HEADER = ["left_ear", "right_ear", "avg_ear", "label"]


def write_records(dataset_records, mode="a"):
    if not dataset_records:
        logger.warning("Khong co mau du lieu nao duoc ghi.")
        return

    os.makedirs(config.DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(config.DATASET_PATH)
    with open(config.DATASET_PATH, mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if mode == "w" or not file_exists or os.path.getsize(config.DATASET_PATH) == 0:
            writer.writerow(CSV_HEADER)
        writer.writerows(dataset_records)

    logger.info("Da ghi %s mau vao %s", len(dataset_records), config.DATASET_PATH)


def generate_synthetic_records(samples_per_class=500):
    records = []
    for _ in range(samples_per_class):
        left_ear = random.uniform(0.29, 0.42)
        right_ear = random.uniform(0.29, 0.42)
        avg_ear = (left_ear + right_ear) / 2.0
        records.append([left_ear, right_ear, avg_ear, 0])

    for _ in range(samples_per_class):
        left_ear = random.uniform(0.08, 0.19)
        right_ear = random.uniform(0.08, 0.19)
        avg_ear = (left_ear + right_ear) / 2.0
        records.append([left_ear, right_ear, avg_ear, 1])

    random.shuffle(records)
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Collect or generate WakeGuard EAR dataset.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic EAR samples instead of using the webcam.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=500,
        help="Number of synthetic samples for each class.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite data/dataset.csv instead of appending.",
    )
    return parser.parse_args()

def draw_instruction(frame, text, text2=None, bg_color=(0, 0, 0)):
    """Draw instructions overlay text on frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 95), bg_color, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(
        frame,
        text,
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if text2:
        cv2.putText(
            frame,
            text2,
            (24, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

def main():
    args = parse_args()
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

    if args.synthetic:
        records = generate_synthetic_records(args.samples_per_class)
        write_records(records, mode="w" if args.overwrite else "a")
        print(f"Da tao {len(records)} mau du lieu gia lap tai: {config.DATASET_PATH}")
        return

    from camera import WebcamSource
    from detector import MediaPipeFaceMeshDetector
    from ear import calculate_avg_ear, calculate_ear
    
    camera = WebcamSource(config.CAMERA_INDEX)
    detector = MediaPipeFaceMeshDetector()
    
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
        
        while camera.is_opened():
            ok, frame = camera.read()
            if not ok or frame is None:
                logger.warning("Khong doc duoc khung hinh tu camera.")
                break

            frame = cv2.flip(frame, 1)
            elapsed = time.time() - state_start_time
            remaining = 0

            if state == 0:
                remaining = max(0, int(prep_duration - elapsed) + 1)
                draw_instruction(
                    frame,
                    f"Chuan bi: MO MAT tu nhien sau {remaining}s",
                    "Nhan Q de thoat.",
                    (30, 70, 30),
                )
                if elapsed >= prep_duration:
                    state = 1
                    state_start_time = time.time()
            elif state == 1:
                remaining = max(0, int(collection_duration - elapsed) + 1)
                draw_instruction(
                    frame,
                    f"Dang thu du lieu MO MAT... {remaining}s",
                    "Giu mat mo binh thuong.",
                    (30, 100, 30),
                )
                landmarks = detector.detect_eye_landmarks(frame)
                if landmarks:
                    left_eye, right_eye = landmarks
                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    avg_ear = calculate_avg_ear(left_eye, right_eye)
                    dataset_records.append([left_ear, right_ear, avg_ear, 0])
                if elapsed >= collection_duration:
                    state = 2
                    state_start_time = time.time()
            elif state == 2:
                remaining = max(0, int(prep_duration - elapsed) + 1)
                draw_instruction(
                    frame,
                    f"Chuan bi: NHAM MAT sau {remaining}s",
                    "Nham mat tu nhien khi bat dau thu.",
                    (70, 60, 20),
                )
                if elapsed >= prep_duration:
                    state = 3
                    state_start_time = time.time()
            elif state == 3:
                remaining = max(0, int(collection_duration - elapsed) + 1)
                draw_instruction(
                    frame,
                    f"Dang thu du lieu NHAM MAT... {remaining}s",
                    "Giu mat nham tu nhien.",
                    (90, 35, 35),
                )
                landmarks = detector.detect_eye_landmarks(frame)
                if landmarks:
                    left_eye, right_eye = landmarks
                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    avg_ear = calculate_avg_ear(left_eye, right_eye)
                    dataset_records.append([left_ear, right_ear, avg_ear, 1])
                if elapsed >= collection_duration:
                    state = 4
            else:
                break

            cv2.putText(
                frame,
                f"Samples: {len(dataset_records)}",
                (24, frame.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("WakeGuard Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                logger.info("Nguoi dung dung thu thap du lieu.")
                break
                
    except Exception as e:
        logger.exception(f"Loi xay ra trong qua trinh thu thap: {e}")
    finally:
        if camera is not None:
            camera.release()
        if detector is not None and hasattr(detector, "close"):
            detector.close()
        cv2.destroyAllWindows()
        
    write_records(dataset_records)

if __name__ == "__main__":
    main()
