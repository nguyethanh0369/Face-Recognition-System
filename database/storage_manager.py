import os
import cv2
import datetime
import logging
import config

logger = logging.getLogger("StorageManager")

class StorageManager:
    def __init__(self):
        """Khởi tạo và tự động tạo các thư mục lưu trữ nếu chưa có sẵn"""
        self.registered_dir = config.REGISTERED_FACES_DIR
        self.logs_dir = config.ATTENDANCE_LOGS_DIR
        
        # Tự sinh thư mục khi chạy ứng dụng
        os.makedirs(self.registered_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        logger.info("Hệ thống lưu trữ ảnh (Local Storage) đã sẵn sàng.")

    def save_registered_face(self, student_id, image_frame):
        """Lưu ảnh gốc lúc sinh viên đăng ký hệ thống"""
        try:
            if image_frame is None: return False
            file_path = os.path.join(self.registered_dir, f"{student_id}.jpg")
            success = cv2.imwrite(file_path, image_frame)
            if success:
                logger.info(f"Đã lưu ảnh gốc đăng ký của SV {student_id} vào local storage.")
                return file_path
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lưu ảnh đăng ký cho SV {student_id}: {e}")
            return None
        
    def save_image(self, student_id, image_frame, is_liveness=1):
        """
        Lưu ảnh chụp từ camera xuống ổ đĩa phục vụ việc đối soát.
        Cấu trúc lưu trữ: captured_faces/YYYY-MM-DD/{student_id}_{Timestamp}_{Real/Fake}.jpg
        """
        try:
            if image_frame is None: return None
            
            # Tự động tạo thư mục con theo ngày bên trong attendance_logs/
            date_str = datetime.date.today().isoformat()
            target_daily_dir = os.path.join(self.logs_dir, date_str)
            os.makedirs(target_daily_dir, exist_ok=True)
            
            # Đặt tên tệp chi tiết
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            status_str = "REAL" if is_liveness == 1 else "FAKE"
            filename = f"{student_id}_{timestamp}_{status_str}.jpg"
            file_path = os.path.join(target_daily_dir, filename)
            
            # Ghi file ảnh bằng OpenCV
            success = cv2.imwrite(file_path, image_frame)
            if success:
                logger.info(f"Đã lưu ảnh đối soát thành công: {file_path}")
                return file_path
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lưu tệp ảnh chụp đối soát: {e}")
            return None

    def get_image_path(self, student_id, date_str, filename_keyword=None):
        """Tìm đường dẫn tệp ảnh dựa vào ID sinh viên và ngày chụp để đối soát."""
        try:
            target_dir = os.path.join(self.logs_dir, date_str)
            if not os.path.exists(target_dir): return None
            
            for file in os.listdir(target_dir):
                if file.startswith(student_id):
                    if filename_keyword and filename_keyword not in file:
                        continue
                    return os.path.join(target_dir, file)
            return None
        except Exception as e:
            logger.error(f"Lỗi khi tra cứu ảnh đối soát: {e}")
            return None

    def auto_cleanup_old_images(self, max_days=None):
        """Tự động xóa các thư mục ảnh cũ vượt quá số ngày quy định để giải phóng dung lượng ổ cứng."""
        try:
            # Ưu tiên lấy số ngày cấu hình trong config.py nếu không truyền tham số riêng
            days_limit = max_days if max_days is not None else getattr(config, 'MAX_STORAGE_DAYS', 30)
            now = datetime.datetime.now()
            count = 0

            if not os.path.exists(self.logs_dir): return count
            
            for folder_name in os.listdir(self.logs_dir):
                folder_path = os.path.join(self.logs_dir, folder_name)
                if os.path.isdir(folder_path):
                    try:
                        folder_date = datetime.datetime.strptime(folder_name, "%Y-%m-%d")
                        age_days = (now - folder_date).days
                        
                        if age_days > days_limit:
                            for file in os.listdir(folder_path):
                                os.remove(os.path.join(folder_path, file))
                            os.rmdir(folder_path)
                            count += 1
                            logger.info(f"Đã xóa thư mục ảnh log hết hạn: {folder_name}")
                    except ValueError:
                        continue # Bỏ qua nếu thư mục không đúng định dạng ngày YYYY-MM-DD
            return count
        except Exception as e:
            logger.error(f"Lỗi trong quá trình dọn dẹp ảnh cũ: {e}")
            return 0