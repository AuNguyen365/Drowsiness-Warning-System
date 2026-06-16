# Tài Liệu Kiến Trúc & Thiết Kế Hệ Thống

Tài liệu này giải thích chi tiết thiết kế bên trong, các công thức toán học và nguyên lý kỹ thuật phần mềm được áp dụng để xây dựng hệ thống **WakeGuard**.

---

## 1. Cơ Sở Toán Học: Tỷ Lệ Khung Hình Mắt (Eye Aspect Ratio - EAR)

Tỷ lệ Khung hình Mắt (EAR) là một thước đo thị giác máy tính cổ điển (được giới thiệu bởi Soukupová và Čech vào năm 2016) dùng để ước tính mức độ mở của mắt từ các tọa độ điểm mốc 2D.

Đối với mỗi khung hình, hệ thống sẽ ánh xạ 6 điểm mốc (landmarks) phác thảo đường viền của mắt:
- $p_1$: Góc ngoài của mắt
- $p_2, p_3$: Điểm mí mắt trên
- $p_4$: Góc trong của mắt
- $p_5, p_6$: Điểm mí mắt dưới

```
       p2     p3
      *-------*
  p1 /         \ p4
    *           *
     \         /
      *-------*
       p6     p5
```

Công thức tính EAR cho một mắt như sau:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \times \|p_1 - p_4\|}$$

- **Tử số** đo khoảng cách mở theo chiều dọc của mắt.
- **Mẫu số** đo chiều rộng theo chiều ngang của mắt, nhằm chuẩn hóa tỷ lệ.
- Nhờ việc chuẩn hóa này, chỉ số EAR ít bị ảnh hưởng bởi tỷ lệ khoảng cách khuôn mặt đến camera.

### Giá Trị EAR Trung Bình
Chúng tôi tính toán chỉ số EAR cho cả mắt trái và mắt phải độc lập, sau đó lấy giá trị trung bình để tăng tính chính xác và chống nhiễu:

$$\text{EAR}_{\text{avg}} = \frac{\text{EAR}_{\text{left}} + \text{EAR}_{\text{right}}}{2}$$

---

## 2. Áp Dụng Các Nguyên Lý SOLID

WakeGuard được thiết kế với mục tiêu dễ mở rộng, dễ kiểm thử và tuân thủ các nguyên lý thiết kế SOLID:

### Nguyên Lý Đơn Nhiệm (Single Responsibility Principle - SRP)
Mỗi lớp trong hệ thống chỉ có một trách nhiệm duy nhất:
- `VideoSource`: Quản lý việc đọc khung hình từ phần cứng camera hoặc luồng tệp.
- `FaceDetector`: Chỉ xác định vùng khuôn mặt và trích xuất tọa độ mắt đã chuẩn hóa.
- `EARCalculator`: Chứa các hàm toán học thuần túy để tính toán các tỷ lệ khoảng cách.
- `AlertSystem`: Quản lý hệ thống cảnh báo (đa luồng, tắt/mở âm thanh).
- `UIService`: Điều khiển vẽ thông tin đồ họa và giao diện HUD lên khung hình.

### Nguyên Lý Đóng/Mở (Open/Closed Principle - OCP)
Các thành phần trong mã nguồn được thiết kế mở để mở rộng nhưng đóng đối với việc sửa đổi. Ví dụ, vòng lặp chính trong `app.py` giao tiếp với hệ thống cảnh báo qua giao diện trừu tượng `AlertSystem`. Nếu trong tương lai chúng ta muốn thêm chức năng gửi tin nhắn SMS cảnh báo, hoặc kích hoạt đèn LED qua cổng GPIO của Raspberry Pi, chúng ta chỉ cần tạo một lớp mới (ví dụ: `IoTAlertSystem(AlertSystem)`) mà không cần thay đổi bất kỳ dòng mã nào bên trong `app.py`.

### Nguyên Lý Thay Thế Liskov (Liskov Substitution Principle - LSP)
Mọi lớp con của `VideoSource` (chẳng hạn như `WebcamSource` hoặc một nguồn kiểm thử video giả lập) đều có thể thay thế hoàn hảo cho vị trí của lớp cha trừu tượng mà không làm phá vỡ hoạt động của chương trình.

### Nguyên Lý Phân Tách Giao Diện (Interface Segregation Principle - ISP)
Các lớp chỉ phụ thuộc vào những phương thức mà chúng thực sự sử dụng. Giao diện như `VideoSource` được thiết kế tối giản (`read()`, `is_opened()`, `release()`) để việc hiện thực hóa các lớp con thay thế trở nên rất đơn giản.

### Nguyên Lý Đảo Ngược Phụ Thuộc (Dependency Inversion Principle - DIP)
Các mô-đun cấp cao trong `app.py` không nhập trực tiếp các lớp cụ thể ở những nơi cần tính đa hình. Chúng dựa vào các giao diện trừu tượng (như `VideoSource` và `AlertSystem`), vốn được tiêm (inject) hoặc khởi tạo một cách lỏng lẻo.

---

## 3. Luồng Dữ Liệu Xử Lý Thời Gian Thực

1. **Đầu vào (Inflow)**: OpenCV chụp một khung hình thô từ webcam bên trong `WebcamSource`.
2. **Phát hiện (Detection)**: Khung hình RGB được đưa vào bộ xử lý MediaPipe FaceMesh bên trong `MediaPipeFaceMeshDetector` để định vị danh sách các tọa độ điểm mốc mắt.
3. **Tính toán (Calculation)**: Các tọa độ điểm được gửi đến hàm `calculate_avg_ear()` trong mô-đun `ear.py` để tính ra một tỷ lệ EAR trung bình duy nhất dạng số thập phân.
4. **Phân tích trạng thái (State Analysis)**: Bộ theo dõi trạng thái sẽ tăng hoặc đặt lại bộ đếm số lượng khung hình liên tiếp nhắm mắt dưới ngưỡng cảnh báo.
5. **Hành động (Action)**:
   - Nếu bộ đếm vượt quá số khung hình quy định, luồng âm thanh cảnh báo sẽ được đánh thức qua hàm `alert_system.start_alarm()`.
   - Nếu EAR tăng trở lại vượt ngưỡng, âm thanh cảnh báo sẽ ngay lập tức được tắt qua hàm `alert_system.stop_alarm()`.
6. **Đầu ra (Outflow)**: Khung hình được vẽ đè các thông số đường viền mắt, trạng thái cảnh báo và thanh tiến trình EAR bởi lớp `UIService` trước khi hiển thị ra màn hình cho người lái xe.
