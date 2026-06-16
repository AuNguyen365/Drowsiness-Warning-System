import os
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)

class DrowsinessClassifier:
    """
    Classifier wrapping the Machine Learning model.
    """
    
    def __init__(self):
        self.model = None
        # TODO: Load the model during initialization
        pass
        
    def _load_model(self) -> None:
        # TODO: Implement model loading logic
        # 1. Check if model file config.MODEL_PATH exists. If not, log a warning and return.
        # 2. Import and use joblib to load the model file, saving it into self.model.
        # 3. Handle import errors or loading exceptions gracefully.
        pass
            
    def predict(self, left_ear: float, right_ear: float, avg_ear: float) -> int:
        """
        Classifies the eye state based on EAR features.
        
        Args:
            left_ear (float): Left eye EAR.
            right_ear (float): Right eye EAR.
            avg_ear (float): Average EAR.
            
        Returns:
            int: 0 for Open, 1 for Closed.
        """
        # TODO: Implement state prediction
        # 1. If self.model is loaded, call predict on model using feature format [[left_ear, right_ear, avg_ear]]
        # 2. Return prediction label (0 or 1)
        # 3. If model is not loaded (or prediction fails), fall back to manual threshold check:
        #    Return 1 if avg_ear < config.EAR_THRESHOLD else 0
        return 0
        
    def is_using_ml(self) -> bool:
        """
        Check if the active classification is driven by ML or heuristic fallback.
        
        Returns:
            bool: True if SVM is running, False otherwise.
        """
        # TODO: Return True if ML model is active, False otherwise
        return False
