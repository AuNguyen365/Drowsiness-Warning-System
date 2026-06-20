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
        self._load_model()
        
    def _load_model(self) -> None:
        if not os.path.exists(config.MODEL_PATH):
            logger.warning("ML model not found at %s. Using EAR threshold fallback.", config.MODEL_PATH)
            return

        try:
            import joblib
        except ImportError:
            logger.warning("joblib is not installed. Using EAR threshold fallback.")
            return

        try:
            self.model = joblib.load(config.MODEL_PATH)
            logger.info("Loaded ML model from %s", config.MODEL_PATH)
        except Exception as e:
            logger.exception("Failed to load ML model from %s: %s", config.MODEL_PATH, e)
            self.model = None
            
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
        if self.model is not None:
            try:
                prediction = self.model.predict([[left_ear, right_ear, avg_ear]])
                return int(prediction[0])
            except Exception as e:
                logger.exception("ML prediction failed, using fallback: %s", e)

        return 1 if avg_ear < config.EAR_THRESHOLD else 0
        
    def is_using_ml(self) -> bool:
        """
        Check if the active classification is driven by ML or heuristic fallback.
        
        Returns:
            bool: True if SVM is running, False otherwise.
        """
        return self.model is not None
