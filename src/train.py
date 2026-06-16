import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("WakeGuard.Trainer")

def main():
    print("=========================================================")
    print("        WAKEGUARD MACHINE LEARNING TRAINING TOOL         ")
    print("=========================================================")
    
    if not os.path.exists(config.DATASET_PATH):
        print(f"Loi: Khong tim thay tep du lieu tai {config.DATASET_PATH}.")
        print("Vui long chay lenh: python src/collect_data.py de thu thap du lieu truoc.")
        return
        
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.metrics import classification_report, accuracy_score
        import joblib
    except ImportError as e:
        print(f"Loi: Thieu thu vien phu thuoc ({e}).")
        print("Vui long dam bao da kich hoat venv va cai dat thanh cong: pip install -r requirements.txt")
        return

    # TODO: Implement model training pipeline:
    # 1. Load CSV data using pandas.
    # 2. Extract features X (left_ear, right_ear, avg_ear) and target y (label).
    # 3. Split dataset into train and test sets (e.g., test_size=0.2, stratify=y).
    # 4. Instantiate a Support Vector Machine classifier (SVC) with an RBF kernel.
    # 5. Fit the classifier on the training data.
    # 6. Evaluate accuracy score and print a detailed classification report on the test set.
    # 7. Ensure config.MODELS_DIR exists, then serialize and export the trained model to config.MODEL_PATH.
    pass

if __name__ == "__main__":
    main()
