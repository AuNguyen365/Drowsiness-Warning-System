import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("WakeGuard.Trainer")

FEATURE_COLUMNS = ["left_ear", "right_ear", "avg_ear"]
TARGET_COLUMN = "label"

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
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError as e:
        print(f"Loi: Thieu thu vien phu thuoc ({e}).")
        print("Vui long dam bao da kich hoat venv va cai dat thanh cong: pip install -r requirements.txt")
        return

    try:
        dataset = pd.read_csv(config.DATASET_PATH)
        missing_columns = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in dataset.columns]
        if missing_columns:
            print(f"Loi: Dataset thieu cot: {', '.join(missing_columns)}")
            return

        dataset = dataset.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
        if dataset.empty:
            print("Loi: Dataset khong co dong du lieu hop le.")
            return

        X = dataset[FEATURE_COLUMNS].astype(float)
        y = dataset[TARGET_COLUMN].astype(int)

        if y.nunique() < 2:
            print("Loi: Dataset can co it nhat 2 nhan (0=mo mat, 1=nham mat).")
            return

        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Do chinh xac: {accuracy:.4f}")
        print("Bao cao phan loai:")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["Open", "Closed"],
                zero_division=0,
            )
        )

        os.makedirs(config.MODELS_DIR, exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)
        print(f"Da luu mo hinh SVM tai: {config.MODEL_PATH}")
    except Exception as e:
        logger.exception(f"Loi khi huan luyen mo hinh: {e}")
        print(f"Loi khi huan luyen mo hinh: {e}")

if __name__ == "__main__":
    main()
