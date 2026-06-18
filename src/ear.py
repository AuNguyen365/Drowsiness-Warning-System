import math
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def calculate_ear(eye_points: List[Tuple[int | float, int | float]]) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for a single eye.
    
    The eye points are assumed to be in order:
    p1 (index 0): Left corner (horizontal)
    p2 (index 1): Top-left (vertical)
    p3 (index 2): Top-right (vertical)
    p4 (index 3): Right corner (horizontal)
    p5 (index 4): Bottom-right (vertical)
    p6 (index 5): Bottom-left (vertical)
    
    Formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
        
    Args:
        eye_points (List[Tuple[int | float, int | float]]): A list of 6 (x, y) coordinates.
        
    Returns:
        float: Calculated Eye Aspect Ratio. Returns 0.0 if calculations are invalid.
    """
    if len(eye_points) < 6:
        logger.warning("Invalid eye points length. EAR requires 6 points.")
        return 0.0
        
    # eye_points are: p1 (0), p2 (1), p3 (2), p4 (3), p5 (4), p6 (5)
    # Euclidean distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)
    def dist(p_a, p_b):
        return math.sqrt((p_a[0] - p_b[0])**2 + (p_a[1] - p_b[1])**2)
        
    d2_6 = dist(eye_points[1], eye_points[5])
    d3_5 = dist(eye_points[2], eye_points[4])
    d1_4 = dist(eye_points[0], eye_points[3])
    
    if d1_4 == 0.0:
        logger.warning("Horizontal eye distance is 0.0. Division by zero prevented.")
        return 0.0
        
    return (d2_6 + d3_5) / (2.0 * d1_4)


def calculate_avg_ear(
    left_eye: List[Tuple[int | float, int | float]], 
    right_eye: List[Tuple[int | float, int | float]]
) -> float:
    """
    Calculate the average Eye Aspect Ratio (EAR) for both eyes.
    
    Args:
        left_eye (List[Tuple[int | float, int | float]]): 6 landmarks of the left eye.
        right_eye (List[Tuple[int | float, int | float]]): 6 landmarks of the right eye.
        
    Returns:
        float: Average EAR of both eyes.
    """
    ear_left = calculate_ear(left_eye)
    ear_right = calculate_ear(right_eye)
    return (ear_left + ear_right) / 2.0
