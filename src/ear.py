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
    # TODO: Calculate Euclidean distances between horizontal and vertical eye landmark pairs.
    # Be sure to handle division by zero errors if horizontal distance is 0.0.
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
    # TODO: Compute individual EAR values for left and right eyes, then return the average.
    return 0.0
