import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# WakeGuard - Mô Phỏng Hệ Thống Cảnh Báo Buồn Ngủ (Drowsiness Warning System)\n",
    "\n",
    "Notebook này mô phỏng toàn bộ quy trình xử lý dữ liệu, phân tích đặc trưng EAR (Eye Aspect Ratio) và huấn luyện mô hình học máy SVM dùng để phát hiện trạng thái ngủ gật của tài xế."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 1. Khai báo các thư viện cần thiết"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
    "from sklearn.pipeline import make_pipeline\n",
    "\n",
    "# Cấu hình biểu đồ\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "%matplotlib inline\n",
    "print(\"Đã import toàn bộ thư viện cần thiết.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 2. Tải và Khám phá Tập dữ liệu (EDA)\n",
    "\n",
    "Chúng ta sẽ đọc tập dữ liệu từ tệp `data/dataset.csv`. Tệp này lưu trữ các đặc trưng EAR của mắt trái, mắt phải và mắt trung bình được trích xuất từ tập ảnh **MRL Eye Dataset**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset_path = os.path.join(\"data\", \"dataset.csv\")\n",
    "\n",
    "if not os.path.exists(dataset_path):\n",
    "    raise FileNotFoundError(f\"Không tìm thấy tệp dataset tại {dataset_path}. Vui lòng kiểm tra lại đường dẫn!\")\n",
    "\n",
    "df = pd.read_csv(dataset_path)\n",
    "print(f\"Tổng số mẫu dữ liệu: {df.shape[0]}\")\n",
    "print(f\"Số cột đặc trưng: {df.shape[1]}\")\n",
    "df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Kiểm tra sự cân bằng của các nhãn (0 = Mở mắt, 1 = Nhắm mắt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "label_counts = df[\"label\"].value_counts()\n",
    "print(\"Phân bố nhãn:\")\n",
    "print(f\"  - Mở mắt (Nhãn 0): {label_counts.get(0, 0)} mẫu\")\n",
    "print(f\"  - Nhắm mắt (Nhãn 1): {label_counts.get(1, 0)} mẫu\")\n",
    "\n",
    "plt.figure(figsize=(6, 4))\n",
    "sns.countplot(x=\"label\", data=df, hue=\"label\", legend=False, palette=\"Set2\")\n",
    "plt.xticks([0, 1], [\"Mở mắt (0)\", \"Nhắm mắt (1)\"])\n",
    "plt.title(\"Biểu đồ phân bố các nhãn trong tập dữ liệu\")\n",
    "plt.xlabel(\"Trạng thái mắt\")\n",
    "plt.ylabel(\"Số lượng mẫu\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Trực quan hóa Phân bố của EAR (Eye Aspect Ratio)\n",
    "\n",
    "EAR là thương số tính từ khoảng cách giữa các điểm mốc mí mắt. Trạng thái mắt mở sẽ có EAR cao (thường từ 0.25 - 0.45), nhắm mắt sẽ có EAR thấp (dưới 0.20)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 6))\n",
    "sns.kdeplot(data=df[df[\"label\"] == 0], x=\"avg_ear\", fill=True, label=\"Mở mắt (0)\", color=\"green\", alpha=0.5)\n",
    "sns.kdeplot(data=df[df[\"label\"] == 1], x=\"avg_ear\", fill=True, label=\"Nhắm mắt (1)\", color=\"red\", alpha=0.5)\n",
    "plt.axvline(x=0.25, color='blue', linestyle='--', label='Ngưỡng mặc định (0.25)')\n",
    "plt.title(\"Phân phối mật độ của chỉ số EAR trung bình (avg_ear)\")\n",
    "plt.xlabel(\"Giá trị EAR\")\n",
    "plt.ylabel(\"Mật độ (Density)\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Qua biểu đồ trên, ta thấy sự phân tách rõ ràng giữa EAR khi mở mắt và nhắm mắt. \n",
    "- Ngưỡng heuristic cố định (0.25) hoạt động tương đối ổn cho đa số trường hợp.\n",
    "- Tuy nhiên, đối với người đeo kính hoặc cấu trúc mắt đặc biệt, các phân phối này có thể dịch chuyển, đó là lý do mô hình học máy (SVM) sẽ giúp tìm ra ranh giới tối ưu hơn."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 3. Huấn luyện Mô hình Học Máy SVM\n",
    "\n",
    "Chúng ta tách tập dữ liệu thành các đặc trưng đầu vào `X` và nhãn đích `y`, phân chia tập Train/Test theo tỉ lệ 80/20, sau đó huấn luyện mô hình SVM sử dụng RBF Kernel."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Lọc các đặc trưng\n",
    "features = [\"left_ear\", \"right_ear\", \"avg_ear\"]\n",
    "X = df[features]\n",
    "y = df[\"label\"]\n",
    "\n",
    "# Chia Train/Test\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "print(f\"Kích thước tập Train: {X_train.shape[0]} mẫu\")\n",
    "print(f\"Kích thước tập Test: {X_test.shape[0]} mẫu\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Xây dựng Pipeline chuẩn hóa và Huấn luyện"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tạo Pipeline: Chuẩn hóa đặc trưng (StandardScaler) -> Phân loại SVM (SVC)\n",
    "pipeline = make_pipeline(StandardScaler(), SVC(kernel=\"rbf\", C=1.0, gamma=\"scale\"))\n",
    "pipeline.fit(X_train, y_train)\n",
    "print(\"Đã huấn luyện xong mô hình SVM.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 4. Đánh giá Mô hình"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dự đoán trên tập kiểm thử\n",
    "y_pred = pipeline.predict(X_test)\n",
    "\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"Độ chính xác toàn cục (Accuracy): {accuracy * 100:.2f}%\")\n",
    "print(\"\\nBáo cáo phân loại (Classification Report):\")\n",
    "print(classification_report(y_test, y_pred, target_names=[\"Mở mắt (Open)\", \"Nhắm mắt (Closed)\"]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Trực quan hóa Ma trận nhầm lẫn (Confusion Matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(y_test, y_pred)\n",
    "plt.figure(figsize=(6, 5))\n",
    "sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", xticklabels=[\"Mở\", \"Nhắm\"], yticklabels=[\"Mở\", \"Nhắm\"])\n",
    "plt.ylabel('Thực tế (Actual)')\n",
    "plt.xlabel('Dự đoán (Predicted)')\n",
    "plt.title('Ma trận nhầm lẫn của mô hình SVM')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 5. Mô Phỏng Quá Trình Nhận Diện Thời Gian Thực\n",
    "\n",
    "Dưới đây là hàm giả lập bộ phân loại trạng thái mắt. Khi camera ghi nhận các đặc trưng mắt của bạn, hệ thống sẽ đưa qua hàm này để dự đoán và ra quyết định cảnh báo nếu nhắm mắt liên tục vượt quá số khung hình quy định."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def simulate_eye_state_detector(left, right):\n",
    "    \"\"\"\n",
    "    Mô phỏng chức năng nhận diện trạng thái mắt dựa trên mô hình SVM đã huấn luyện.\n",
    "    \"\"\"\n",
    "    avg = (left + right) / 2.0\n",
    "    # Tạo DataFrame để khớp với tên đặc trưng\n",
    "    input_data = pd.DataFrame([[left, right, avg]], columns=[\"left_ear\", \"right_ear\", \"avg_ear\"])\n",
    "    \n",
    "    # Dự đoán\n",
    "    pred = pipeline.predict(input_data)[0]\n",
    "    \n",
    "    state = \"NHẮM MẮT (Closed)\" if pred == 1 else \"MỞ MẮT (Open)\"\n",
    "    print(f\"Chỉ số nhận diện: Left={left:.3f}, Right={right:.3f}, Average={avg:.3f}\")\n",
    "    print(f\"=> Trạng thái dự đoán của mô hình: {state}\\n\")\n",
    "    return pred\n",
    "\n",
    "# Kiểm thử các trường hợp khác nhau:\n",
    "print(\"--- Thử nghiệm 1: Mắt mở to bình thường ---\")\n",
    "simulate_eye_state_detector(0.38, 0.36)\n",
    "\n",
    "print(\"--- Thử nghiệm 2: Mắt nhắm chặt ---\")\n",
    "simulate_eye_state_detector(0.12, 0.11)\n",
    "\n",
    "print(\"--- Thử nghiệm 3: Trạng thái lờ đờ / mỏi mắt / đeo kính bị lệch chỉ số ---\")\n",
    "simulate_eye_state_detector(0.22, 0.23)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 6. Mô Phỏng Bộ Đếm Khung Hình Cảnh Báo (Drowsiness Alert Logic)\n",
    "\n",
    "Hệ thống sẽ không cảnh báo ngay lập tức khi bạn chớp mắt (nhắm mắt 1-2 khung hình). Cảnh báo chỉ kích hoạt khi bạn nhắm mắt liên tục vượt quá `CONSECUTIVE_FRAMES` (mặc định là 15 khung hình, tương đương khoảng 0.5 giây)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Giả lập chuỗi trạng thái mắt đọc được từ camera qua thời gian\n",
    "# 0 = Mở, 1 = Nhắm\n",
    "simulated_stream = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]\n",
    "\n",
    "consec_max = 15\n",
    "counter = 0\n",
    "alarm_active = False\n",
    "\n",
    "print(f\"Bắt đầu mô phỏng luồng stream với ngưỡng khung hình nhắm mắt liên tục = {consec_max}\\n\")\n",
    "for frame_idx, state in enumerate(simulated_stream):\n",
    "    if state == 1:\n",
    "        counter += 1\n",
    "        status = f\"[Nhắm mắt] Bộ đếm: {counter}/{consec_max}\"\n",
    "        if counter >= consec_max:\n",
    "            if not alarm_active:\n",
    "                alarm_active = True\n",
    "                status += \" -> !!! KÍCH HOẠT CÒI BÁO ĐỘNG (ALARM ON) !!!\"\n",
    "            else:\n",
    "                status += \" -> [Còi vẫn kêu]\"\n",
    "    else:\n",
    "        counter = 0\n",
    "        status = \"[Mở mắt] Bộ đếm reset về 0\"\n",
    "        if alarm_active:\n",
    "            alarm_active = False\n",
    "            status += \" -> Tắt còi báo động (ALARM OFF)\"\n",
    "            \n",
    "    print(f\"Khung hình #{frame_idx + 1:02d}: {status}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs("scratch", exist_ok=True)
with open("drowsiness_detection_simulation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Created drowsiness_detection_simulation.ipynb successfully.")
