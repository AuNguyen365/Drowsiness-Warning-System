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
    if len(eye_points) != 6:
        logger.error(f"EAR calculation requires exactly 6 points. Received {len(eye_points)}.")
        return 0.0
        
    try:
        # Extract coordinates
        p1, p2, p3, p4, p5, p6 = eye_points
        
        # Calculate Euclidean distances for vertical pairs
        # ||p2 - p6||
        dist_p2_p6 = math.dist(p2, p6)
        # ||p3 - p5||
        dist_p3_p5 = math.dist(p3, p5)
        
        # Calculate Euclidean distance for horizontal pair
        # ||p1 - p4||
        dist_p1_p4 = math.dist(p1, p4)
        
        # Guard against division by zero
        if dist_p1_p4 == 0.0:
            logger.warning("Horizontal distance of eye is zero. Returning EAR = 0.0 to prevent division by zero.")
            return 0.0
            
        # Compute EAR
        ear = (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)
        return ear
        
    except Exception as e:
        logger.error(f"Error during EAR calculation: {e}")
        return 0.0


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
