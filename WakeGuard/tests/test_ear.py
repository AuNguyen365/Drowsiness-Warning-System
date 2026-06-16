import os
import sys
import math
import pytest

# Add the src folder to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ear import calculate_ear, calculate_avg_ear

def test_calculate_ear_standard():
    # Construct an artificial open eye:
    # Width (horizontal distance between p1 and p4) = 10
    # Heights (vertical distances between p2-p6 and p3-p5) = 5 and 5
    # p1 = (0, 0), p2 = (3, 2.5), p3 = (7, 2.5), p4 = (10, 0), p5 = (7, -2.5), p6 = (3, -2.5)
    # dist(p1, p4) = 10
    # dist(p2, p6) = dist((3, 2.5), (3, -2.5)) = 5
    # dist(p3, p5) = dist((7, 2.5), (7, -2.5)) = 5
    # EAR = (5 + 5) / (2 * 10) = 10 / 20 = 0.50
    eye_pts = [
        (0, 0),       # p1
        (3, 2.5),     # p2
        (7, 2.5),     # p3
        (10, 0),      # p4
        (7, -2.5),    # p5
        (3, -2.5)     # p6
    ]
    ear = calculate_ear(eye_pts)
    assert pytest.approx(ear) == 0.50

def test_calculate_ear_closed():
    # Construct a closed eye (vertical distance collapsed to 0)
    # p1=(0,0), p2=(3,0), p3=(7,0), p4=(10,0), p5=(7,0), p6=(3,0)
    # dist(p1, p4) = 10
    # dist(p2, p6) = 0
    # dist(p3, p5) = 0
    # EAR = (0 + 0) / (2 * 10) = 0.0
    eye_pts = [
        (0, 0),
        (3, 0),
        (7, 0),
        (10, 0),
        (7, 0),
        (3, 0)
    ]
    ear = calculate_ear(eye_pts)
    assert ear == 0.0

def test_calculate_ear_invalid_length():
    # Less than 6 coordinates should return 0.0 safely without raising an error
    eye_pts = [(0, 0), (1, 1)]
    ear = calculate_ear(eye_pts)
    assert ear == 0.0

def test_calculate_ear_division_by_zero():
    # Horizontal distance is zero: p1 == p4
    # All points are at (0, 0)
    eye_pts = [(0, 0)] * 6
    ear = calculate_ear(eye_pts)
    assert ear == 0.0

def test_calculate_avg_ear():
    # Left eye EAR = 0.3, Right eye EAR = 0.5 -> Avg = 0.4
    left_eye = [
        (0, 0),
        (3, 1.5), # dist p2-p6 = 3
        (7, 1.5), # dist p3-p5 = 3
        (10, 0),  # dist p1-p4 = 10
        (7, -1.5),
        (3, -1.5)
    ] # EAR = (3 + 3) / 20 = 0.3
    
    right_eye = [
        (0, 0),
        (3, 2.5), # dist p2-p6 = 5
        (7, 2.5), # dist p3-p5 = 5
        (10, 0),  # dist p1-p4 = 10
        (7, -2.5),
        (3, -2.5)
    ] # EAR = (5 + 5) / 20 = 0.5
    
    avg_ear = calculate_avg_ear(left_eye, right_eye)
    assert pytest.approx(avg_ear) == 0.40
