# Phân Công Nhiệm Vụ & Phương Pháp Phát Triển Song Song (WakeGuard)

Bản kế hoạch phân chia công việc chi tiết cho 5 thành viên nhóm phát triển phần mềm WakeGuard. Dự án được triển khai song song bằng cách áp dụng lập trình dựa trên giao diện (Interface-driven), tạo lớp giả lập (Mock/Stub) và dữ liệu giả lập (Synthetic Data).

---

## NGUYÊN TẮC PHÁT TRIỂN SONG SONG (INTERFACE-DRIVEN & MOCKING)

Để không ai phải đợi ai, dự án áp dụng các nguyên tắc sau:
1. **Lập trình hướng giao diện**: Các lớp trừu tượng (như `VideoSource`, `FaceDetector`, `AudioAlertSystem`) đã được định nghĩa sẵn. Thành viên chỉ cần cài đặt phần xử lý bên trong.
2. **Sử dụng Dữ liệu Giả lập (Synthetic Data)**: Thành viên làm AI có thể tự sinh dữ liệu EAR giả lập để huấn luyện mô hình SVM mà không cần chờ luồng camera hay luồng thu thập dữ liệu hoàn thành.
3. **Sử dụng Mock/Stub cho kiểm thử**: Thành viên làm UI và Tích hợp có thể giả lập (mock) các mốc khuôn mặt và trạng thái nhắm mắt để kiểm tra giao diện và luồng chạy chính mà không cần thuật toán thật.

---

## BẢNG TỔNG HỢP PHÂN CÔNG NHIỆM VỤ

| Thành viên | Vai trò kĩ thuật | Mã nguồn phụ trách | Cách thức thực hiện |
| :--- | :--- | :--- | :--- |
| **Thành viên 1** | **Core Algorithm & Face Mesh Engineer** | [detector.py](../src/detector.py)<br>[ear.py](../src/ear.py)<br>[test_ear.py](../tests/test_ear.py) | Sử dụng ảnh/video tĩnh có sẵn trong thư mục `docs/` hoặc `assets/` làm đầu vào để test thuật toán landmark và tính EAR. |
| **Thành viên 2** | **Machine Learning & Data Pipeline Engineer** | [collect_data.py](../src/collect_data.py)<br>[train.py](../src/train.py)<br>[classifier.py](../src/classifier.py) | Tự sinh dữ liệu EAR giả lập (synthetic data) lưu thành `data/dataset.csv` để code và chạy thử quy trình train SVM. |
| **Thành viên 3** | **GUI & HUD Overlay Developer** | [ui.py](../src/ui.py) | Viết script chạy thử UI bằng cách nạp tọa độ mắt giả lập (mock landmarks) và giá trị EAR biến thiên dạng hình sin. |
| **Thành viên 4** | **System Integrator & Stream Controller** | [camera.py](../src/camera.py)<br>[app.py](../src/app.py)<br>[config.py](../src/config.py) | Tích hợp hệ thống bằng cách gọi các lớp giả lập (stubs) hoạt động tối giản của các thành viên khác (ví dụ: detector luôn trả về mắt mở). |
| **Thành viên 5** | **Performance & Async Sound Engineer** | [alert.py](../src/alert.py)<br>Tối ưu luồng song song | Viết hàm main tạm thời trong `alert.py` tự kích hoạt còi báo động lặp lại theo chu kỳ 2 giây để test tính bất đồng bộ. |

---

## MÔ TẢ CHI TIẾT & PHƯƠNG PHÁP THỰC HIỆN

### 1. Thành viên 1: Core Algorithm & Face Mesh Engineer
* **Nhiệm vụ chi tiết**:
  * Khởi tạo `MediaPipeFaceMeshDetector` trong `detector.py`.
  * Trích xuất tọa độ 6 điểm mốc của mắt trái và 6 điểm mốc của mắt phải.
  * Lập trình công thức toán học tính chỉ số EAR trong `ear.py`.
  * Viết mã nguồn kiểm thử đơn vị `test_ear.py` bằng `pytest`.
* **Cách thức thực hiện**:
  * Sử dụng một vài bức ảnh khuôn mặt mắt nhắm và mắt mở đặt trong thư mục `assets/` hoặc dùng ảnh mẫu có sẵn làm đầu vào.
  * Viết một file chạy thử ngắn đọc ảnh tĩnh này qua `cv2.imread()`, truyền qua lớp `detector` và in ra kết quả EAR để tối ưu thuật toán mà không cần quan tâm đến giao diện hay camera stream.

### 2. Thành viên 2: Machine Learning & Data Pipeline Engineer
* **Nhiệm vụ chi tiết**:
  * Lập trình logic ghi nhận EAR và ghi ra file trong `collect_data.py`.
  * Lập trình huấn luyện mô hình SVM, chuẩn hóa dữ liệu và xuất file `.pkl` trong `train.py`.
  * Lập trình lớp bộ lọc `DrowsinessClassifier` trong `classifier.py`.
* **Cách thức thực hiện**:
  * Tạo một script nhỏ để sinh dữ liệu giả lập (synthetic data) gồm 500 dòng EAR > 0.28 (nhãn `1` - mở mắt) và 500 dòng EAR < 0.20 (nhãn `0` - nhắm mắt) lưu vào `data/dataset.csv`.
  * Dùng file CSV giả lập này để code và chạy hoàn thiện pipeline của `train.py`, tạo ra file `models/drowsiness_svm.pkl`.
  * Viết test cho `classifier.py` bằng cách truyền các giá trị EAR tĩnh (ví dụ: `0.15` -> nhắm mắt, `0.32` -> mở mắt) để kiểm tra kết quả dự đoán của mô hình.

### 3. Thành viên 3: GUI & HUD Overlay Developer
* **Nhiệm vụ chi tiết**:
  * Vẽ đồ họa khung viền mắt và các điểm chấm mí mắt lên khung hình camera trong `ui.py`.
  * Thiết kế thanh tiến trình (progress bar) thể hiện mức EAR ở cạnh phải màn hình.
  * Thiết kế banner nhấp nháy đỏ/vàng khi phát hiện buồn ngủ và nhãn ghi thông số hệ thống.
* **Cách thức thực hiện**:
  * Tự định nghĩa một danh sách tọa độ mắt giả lập cố định (ví dụ: `[(100, 100), (110, 90)...]`).
  * Sử dụng một vòng lặp đọc webcam đơn giản hoặc đọc một video clip bất kỳ trên máy tính.
  * Truyền các giá trị EAR tăng giảm nhân tạo (dạng sóng hình sin hoặc tăng dần từ 0.1 đến 0.4) vào lớp UI để test hiển thị của thanh progress bar và banner nhấp nháy mà không cần AI hay MediaPipe.

### 4. Thành viên 4: System Integrator & Stream Controller
* **Nhiệm vụ chi tiết**:
  * Phát triển lớp `WebcamSource` trong `camera.py` bọc `cv2.VideoCapture` để đọc camera.
  * Xây dựng vòng lặp quản lý ứng dụng chính trong `app.py` và tập hợp cấu hình tại `config.py`.
  * Quản lý giải phóng tài nguyên hệ thống khi nhấn nút `Q`.
* **Cách thức thực hiện**:
  * Sử dụng các lớp giả lập tối giản (Stub classes) ngay trong `app.py`. Ví dụ:
    * Tạo lớp `DummyDetector` luôn trả về tọa độ mắt giả lập cố định.
    * Tạo lớp `DummyClassifier` luôn báo tài xế tỉnh táo (hoặc giả lập buồn ngủ sau mỗi 100 khung hình).
    * Tạo lớp `DummyUI` chỉ hiển thị khung hình thô của camera.
  * Tiến hành cấu trúc hóa và chạy thử vòng lặp của `app.py` để kiểm tra độ trễ luồng, cấu trúc kết nối và phím tắt thoát chương trình.

### 5. Thành viên 5: Performance & Async Sound Engineer
* **Mục tiêu**: Lập trình bộ còi báo động bất đồng bộ trong `alert.py` và tối ưu hóa FPS của hệ thống.
* **Cách thức thực hiện**:
  * Viết đoạn mã chạy thử nghiệm trực tiếp dưới dạng khối `if __name__ == "__main__":` trong file [alert.py](../src/alert.py).
  * Trong khối chạy thử này, tạo một vòng lặp kích hoạt báo động trong 3 giây, sau đó tắt trong 3 giây, rồi lại bật.
  * Chạy trực tiếp file `alert.py` để kiểm tra âm thanh phát ra (sử dụng `winsound` hoặc `playsound`) có bị ngắt quãng, hoạt động đúng trên luồng phụ (daemon thread) và không gây treo luồng điều phối chính hay không.
