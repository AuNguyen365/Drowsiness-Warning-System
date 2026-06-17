# WakeGuard - Hệ Thống Cảnh Báo Buồn Ngủ Cho Tài Xế

WakeGuard là một hệ thống phát hiện buồn ngủ của tài xế theo thời gian thực được phát triển bằng ngôn ngữ Python. Hệ thống phân tích luồng camera trực tiếp của tài xế bằng các kỹ thuật Thị giác Máy tính (Computer Vision) để theo dõi và đo lường trạng thái mở mắt thông qua chỉ số Tỷ lệ Khung hình Mắt (Eye Aspect Ratio - EAR). 

Hệ thống hỗ trợ cả chế độ phân loại hình học tĩnh và **chế độ Học máy thông minh (Machine Learning - SVM)** để tự động nhận dạng trạng thái mắt mở/nhắm phù hợp với sinh lý mắt của từng người dùng cụ thể.

---

## Tính Năng

- **Phát Hiện Mốc Khuôn Mặt Thời Gian Thực**: Sử dụng thư viện MediaPipe FaceMesh để theo dõi tọa độ khuôn mặt 3D với độ chính xác cao từ webcam tiêu chuẩn.
- **Tỷ Lệ Khung Hình Mắt (EAR)**: Tính toán mức độ mở mắt một cách xác định (deterministic) mà không cần sử dụng mô hình phân loại Deep Learning phức tạp.
- **Hỗ Trợ Học Máy SVM (Mới)**: Huấn luyện mô hình phân loại nhắm/mở mắt cá nhân hóa bằng thuật toán SVM (Scikit-Learn) dựa trên dữ liệu số ghi nhận trực tiếp từ webcam của bạn.
- **Giao Diện Cảnh Báo Trực Quan (HUD)**: Cửa sổ hiển thị video thời gian thực với các đường viền tùy chỉnh bao quanh mắt, chỉ số theo dõi EAR, biểu ngữ cảnh báo nhấp nháy, cột đo mức EAR và nhãn hiển thị chế độ hoạt động hiện tại (SVM hay Tĩnh).
- **Cảnh Báo Âm Thanh Không Gây Nghẽn (Non-blocking)**: Sử dụng luồng chạy ngầm hoặc API hệ thống bất đồng bộ độc lập (`winsound` trên Windows, `playsound` trên các nền tảng khác) để phát âm thanh cảnh báo, đảm bảo không gây giật lag hoặc giảm khung hình (FPS) của luồng xử lý chính.
- **Cấu Hình Linh Hoạt**: Các ngưỡng nhạy cảm EAR và thời gian nhắm mắt có thể dễ dàng tùy chỉnh trong một tệp cấu hình duy nhất.
- **Kiến Trúc SOLID**: Được thiết kế hướng đối tượng, phân tách rõ ràng các thành phần chức năng theo nguyên lý Single Responsibility (Đơn nhiệm) và Dependency Inversion (Đảo ngược Phụ thuộc).

---

## Cấu Trúc Thư Mục Dự Án

```
Drowsiness-Warning-System/ (Thư mục gốc workspace)
├── assets/             # Tài nguyên (ví dụ: tệp âm thanh cảnh báo)
├── data/               # Chứa dữ liệu thu thập (dataset.csv)
├── models/             # Chứa mô hình SVM đã huấn luyện (drowsiness_svm.pkl)
├── docs/               # Tài liệu hệ thống và hướng dẫn chi tiết
├── src/                # Mã nguồn hệ thống
│   ├── alert.py        # Mô-đun cảnh báo âm thanh bất đồng bộ
│   ├── app.py          # Vòng lặp chính điều phối hoạt động hệ thống
│   ├── camera.py       # Lớp trừu tượng hóa việc thu nhận hình ảnh từ camera
│   ├── classifier.py   # Lớp bọc bộ phân loại trạng thái mắt (SVM hoặc Fallback)
│   ├── collect_data.py # Tập lệnh thu thập dữ liệu EAR nhắm/mở mắt
│   ├── config.py       # Các tham số tinh chỉnh cấu hình hệ thống
│   ├── detector.py     # Bộ định vị tọa độ mắt qua MediaPipe FaceMesh
│   ├── ear.py          # Tính toán toán học cho Tỷ lệ Khung hình Mắt (EAR)
│   ├── train.py        # Tập lệnh huấn luyện mô hình SVM từ dữ liệu đã thu thập
│   └── ui.py           # Giao diện HUD vẽ đè thông tin hiển thị (OpenCV)
├── tests/              # Thư mục kiểm thử (Pytest)
│   └── test_ear.py     # Các bài kiểm thử isolated cho thuật toán EAR
└── requirements.txt    # Danh sách các thư viện phụ thuộc của dự án
```

---

## Cài Đặt & Thiết Lập

### Yêu Cầu Hệ Thống
- Python 3.11 trở lên.
- Một webcam hoạt động bình thường kết nối với máy tính.

### 1. Thiết Lập Môi Trường Ảo
Khuyên dùng môi trường ảo Python (virtual environment) để cô lập các thư viện của dự án:
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên Linux/macOS:
source venv/bin/activate
```

### 2. Cài Đặt Thư Viện Phụ Thuộc
Cài đặt các gói thư viện cần thiết bằng pip:
```bash
pip install -r requirements.txt
```

---

## Quy Trình Vận Hành Học Máy (Machine Learning Workflow)

Để kích hoạt chế độ Học máy cá nhân hóa, bạn hãy thực hiện theo 3 bước cực kỳ đơn giản sau:

### Bước 1: Thu thập dữ liệu mắt của chính bạn
Chạy tập lệnh thu thập dữ liệu tự động:
```bash
python src/collect_data.py
```
*Hệ thống sẽ mở camera và hiển thị đếm ngược hướng dẫn bạn:*
- Giữ mắt **Mở** bình thường trong 10 giây.
- **Nhắm mắt** tự nhiên trong 10 giây.
- Dữ liệu EAR thô thu được sẽ tự động lưu vào tệp `data/dataset.csv`.

### Bước 2: Huấn luyện mô hình SVM
Chạy tập lệnh huấn luyện:
```bash
python src/train.py
```
*Tập lệnh sẽ đọc tệp CSV, huấn luyện mô hình SVM và in báo cáo độ chính xác ra màn hình. Mô hình hoàn chỉnh được lưu thành tệp tĩnh tại `models/drowsiness_svm.pkl`.*

### Bước 3: Chạy ứng dụng giám sát thời gian thực
Khởi chạy hệ thống cảnh báo:
```bash
python src/app.py
```
*Hệ thống sẽ tự động phát hiện tệp mô hình `.pkl` và hiển thị trạng thái `MODE: SVM (ML)` màu vàng ở góc dưới giao diện. Nếu chưa huấn luyện mô hình, hệ thống sẽ chạy ở chế độ tĩnh dự phòng `MODE: Static (Fallback)`.*

---

## Các Phím Điều Khiển Trực Tiếp:
* **Nhấn phím `Q` hoặc `q`**: Đóng ứng dụng một cách an toàn, giải phóng tài nguyên camera, dọn dẹp các luồng phát âm thanh ngầm và thoát chương trình.

---

## Cấu Hình Hệ Thống

Tất cả các thiết lập cốt lõi nằm trong tệp [src/config.py](src/config.py). Bạn có thể điều chỉnh trực tiếp các giá trị này để hiệu chuẩn hệ thống:

| Tham số | Kiểu dữ liệu | Giá trị mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `EAR_THRESHOLD` | `float` | `0.25` | Ngưỡng giá trị EAR dùng khi chạy chế độ dự phòng Static. |
| `CONSECUTIVE_FRAMES` | `int` | `15` | Số lượng khung hình nhắm mắt liên tiếp trước khi kích hoạt còi báo động. |
| `CAMERA_INDEX` | `int/str`| `0` | Cổng camera hoặc đường dẫn tệp video cục bộ để thử nghiệm. |
| `ALARM_SOUND_PATH` | `str` | `assets/alarm.wav` | Đường dẫn tới tệp âm thanh cảnh báo (tự động phát tiếng bíp nếu thiếu tệp). |

---

## Xác Minh & Kiểm Thử

### Chạy Kiểm Thử Đơn Vị (Unit Tests)
Để kiểm tra tính toàn vẹn của mã toán học cốt lõi:
```bash
pytest tests/
```

---

## 🔧 Quy định Commit (Git Commit Convention)

Để dự án được quản lý chuyên nghiệp, tất cả thành viên hãy tuân thủ cấu trúc Commit sau:

Cấu trúc: `<type>(<scope>): <description>`

- `feat`: Thêm tính năng mới (Ví dụ: `feat(auth): add google login`)
- `fix`: Sửa lỗi (Ví dụ: `fix(cart): fix price calculation error`)
- `docs`: Cập nhật tài liệu (Ví dụ: `docs(readme): update run instructions`)
- `style`: Thay đổi giao diện, CSS, format code (Ví dụ: `style(home): update hero section colors`)
- `refactor`: Tái cấu trúc mã nguồn, không làm thay đổi tính năng (Ví dụ: `refactor(db): change model structure`)
- `chore`: Các công việc phụ trợ, cài đặt môi trường (Ví dụ: `chore(git): add gitkeep files`)

---

## 📂 Tài Liệu Dự Án

Các tài liệu hướng dẫn và thiết kế chi tiết nằm trong thư mục `docs/`:
- 📋 **[Bảng Phân Công Nhiệm Vụ (docs/Divisiontasks.md)](docs/Divisiontasks.md)**: Phân chia chi tiết nhiệm vụ và cách thức thực hiện cho 5 thành viên.
- 📐 **[Tài Liệu Kiến Trúc Hệ Thống (docs/architecture.md)](docs/architecture.md)**: Mô tả thiết kế kiến trúc hệ thống, sơ đồ luồng dữ liệu và thiết kế lớp hướng đối tượng.

