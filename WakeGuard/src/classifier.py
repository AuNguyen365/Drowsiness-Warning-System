import os
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)

class DrowsinessClassifier:
    """
    Classifier wrapping the Machine Learning model.
    Follows Dependency Inversion and Open/Closed Principles.
    """
    
    def __init__(self):
        self.model = None
        self._load_model()
        
    def _load_model(self) -> None:
        if not os.path.exists(config.MODEL_PATH):
            logger.warning(
                f"Model file not found at '{config.MODEL_PATH}'. "
                f"Fallback to static threshold (EAR < {config.EAR_THRESHOLD}) is active."
            )
            return
            
        try:
            import joblib
            self.model = joblib.load(config.MODEL_PATH)
            logger.info("Successfully loaded trained SVM model for drowsiness detection.")
        except ImportError:
            logger.warning("joblib or scikit-learn is not installed. Falling back to static threshold.")
        except Exception as e:
            logger.error(f"Error loading SVM model: {e}. Falling back to static threshold.")
            
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
                # Features shape: [1, 3]
                prediction = self.model.predict([[left_ear, right_ear, avg_ear]])
                return int(prediction[0])
            except Exception as e:
                logger.error(f"Prediction failed: {e}. Falling back to static threshold.")
                
        # Heuristic Fallback
        return 1 if avg_ear < config.EAR_THRESHOLD else 0
        
    def is_using_ml(self) -> bool:
        """
        Check if the active classification is driven by ML or heuristic fallback.
        
        Returns:
            bool: True if SVM is running, False otherwise.
        """
        return self.model is not None
