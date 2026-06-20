import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import math
import time
from src.ui import UIService


def make_eye(cx, cy, w=40, h=18):
    # Return 6-point eye contour roughly around center
    return [
        (int(cx - w // 2), int(cy)),
        (int(cx - w // 4), int(cy - h // 2)),
        (int(cx + w // 4), int(cy - h // 2)),
        (int(cx + w // 2), int(cy)),
        (int(cx + w // 4), int(cy + h // 2)),
        (int(cx - w // 4), int(cy + h // 2)),
    ]


def main():
    w, h = 960, 540
    ui = UIService(ear_threshold=0.20)
    start = time.time()
    consec = 0
    consec_max = 50

    print("Press 'q' or ESC to quit. Watch for flashing WARNING/DROWSY banners.")
    while True:
        t = time.time() - start
        # Sinusoidal EAR between ~0.12 and ~0.32
        ear = 0.22 + 0.1 * math.sin(2 * math.pi * 0.5 * t)

        # simple drowsy logic
        if ear < ui.ear_threshold:
            consec += 1
        else:
            consec = 0
        is_drowsy = consec > 25

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Add a soft background gradient for visibility
        for i in range(h):
            color = 12 + int(80 * (i / h))
            frame[i, :, :] = (color, color + 10, color + 30)

        # Mock eye positions
        left = make_eye(w * 0.33, h * 0.45, w=70, h=28)
        right = make_eye(w * 0.66, h * 0.45, w=70, h=28)

        # Slight animate eye by moving vertical landmarks based on EAR
        # we approximate vertical displacement from EAR value
        disp = int((0.22 - ear) * 60)
        if disp != 0:
            # move upper points down and lower points up slightly
            left[1] = (left[1][0], left[1][1] + disp)
            left[2] = (left[2][0], left[2][1] + disp)
            left[4] = (left[4][0], left[4][1] - disp)
            left[5] = (left[5][0], left[5][1] - disp)

            right[1] = (right[1][0], right[1][1] + disp)
            right[2] = (right[2][0], right[2][1] + disp)
            right[4] = (right[4][0], right[4][1] - disp)
            right[5] = (right[5][0], right[5][1] - disp)

        out = ui.draw_hud(frame, left, right, ear, is_drowsy, consec, consec_max)

        cv2.imshow("UI Mock", out)
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
