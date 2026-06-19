import os
import sys
import logging
import time
import cv2

# Thêm thư mục src vào path để import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from camera import WebcamSource

def test_camera():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("TestCamera")
    
    logger.info("Khởi tạo WebcamSource với index 0...")
    camera = WebcamSource(camera_index=0)
    
    if not camera.is_opened():
        logger.error("Không thể mở camera index 0. Thử nghiệm camera_index=1 hoặc đường dẫn video nếu có.")
        return
        
    logger.info("Camera đã mở thành công. Tiến hành đọc 30 khung hình...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while frame_count < 30:
            success, frame = camera.read()
            if not success or frame is None:
                logger.warning(f"Lỗi đọc khung hình tại index {frame_count}")
                break
                
            frame_count += 1
            h, w, c = frame.shape
            logger.info(f"Đọc thành công khung hình #{frame_count}: Kích thước {w}x{h}, Channels: {c}")
            
            # Hiển thị thử nghiệm lên màn hình để xác nhận trực quan
            cv2.imshow("Test WebcamSource", frame)
            
            # Đợi 30ms (khoảng 30fps)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                logger.info("Nhấn Q để thoát sớm.")
                break
                
    except Exception as e:
        logger.exception(f"Lỗi trong quá trình đọc camera: {e}")
    finally:
        logger.info("Đang giải phóng tài nguyên camera...")
        camera.release()
        cv2.destroyAllWindows()
        logger.info(f"Hoàn thành kiểm thử. Tổng số khung hình đọc được: {frame_count}")

if __name__ == "__main__":
    test_camera()
