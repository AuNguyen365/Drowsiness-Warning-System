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

    logger.info(f"Dang nap tap du lieu: {config.DATASET_PATH}")
    try:
        df = pd.read_csv(config.DATASET_PATH)
    except Exception as e:
        logger.error(f"Loi nap tep CSV: {e}")
        return
        
    # Validation of headers
    required_cols = {'left_ear', 'right_ear', 'avg_ear', 'label'}
    if not required_cols.issubset(df.columns):
        logger.critical(f"Tep CSV khong dung dinh dang. Can co cac cot: {required_cols}")
        return
        
    logger.info(f"Tong so mau thu thap: {len(df)}")
    print(df['label'].value_counts().rename(index={0: "Mat Mo (Label 0)", 1: "Mat Nham (Label 1)"}))
    
    # Extract features and target
    X = df[['left_ear', 'right_ear', 'avg_ear']]
    y = df['label']
    
    # Split training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info("Dang huan luyen mo hinh Support Vector Machine (SVM)...")
    # Using RBF kernel which works perfectly for classification of non-linear decision boundary
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- KET QUA DANH GIA MO HINH ---")
    print(f"Do chinh xac (Accuracy): {accuracy * 100:.2f}%")
    print("\nBao cao phan loai chi tiet (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=["Mat Mo", "Mat Nham"]))
    
    # Save model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    logger.info(f"Dang xuat mo hinh ra tep: {config.MODEL_PATH}")
    try:
        joblib.dump(clf, config.MODEL_PATH)
        print("\n=========================================================")
        print("Huan luyen thanh cong! Mo hinh da duoc luu.")
        print("Bay gio ban co the khoi chay ung dung canh bao buon ngu bang:")
        print("  python src/app.py")
        print("=========================================================")
    except Exception as e:
        logger.error(f"Loi xuat tep pkl: {e}")

if __name__ == "__main__":
    main()

