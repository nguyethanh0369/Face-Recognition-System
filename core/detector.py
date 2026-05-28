import cv2
import logging
import config

logger = logging.getLogger("FaceEngine.Detector")

class FrameSkipper:
    def __init__(self, skip_interval=None):
        """
        Quản lý việc bỏ qua khung hình để giảm tải cho CPU/GPU.
        Nếu skip_interval không được truyền vào, hệ thống sẽ tự động lấy từ config.py
        """
        if skip_interval is None:
            skip_interval = config.SKIP_INTERVAL

        self.skip_interval = max(1, skip_interval)
        self.frame_count = 0

    def should_process(self) -> bool:
        self.frame_count += 1
        return (self.frame_count - 1) % self.skip_interval == 0

    def reset(self):
        self.frame_count = 0


class FaceDetector:
    def __init__(self, model_path, det_threshold=0.75):
        """
        Quản lý mô hình phát hiện khuôn mặt YuNet
        """
        self.model_path = model_path if model_path is not None else config.YUNET_WEIGHTS
        self.det_threshold = det_threshold if det_threshold is not None else config.DET_THRESHOLD

        try:
            # Khởi tạo mô hình YuNet từ OpenCV với cấu hình chuẩn
            # Tham số cuối cùng (0.3) là NMS Threshold (ngưỡng lọc bớt các hộp trùng lặp)
            self.detector = cv2.FaceDetectorYN.create(
                self.model_path, 
                "", 
                (320, 320), 
                self.det_threshold, 
                0.3
            )
            logger.info(f"Khởi tạo mô hình YuNet thành công. Trọng số: {self.model_path} | Ngưỡng Det: {self.det_threshold}")
        except Exception as e:
            logger.error(f"Không thể khởi tạo mô hình YuNet: {e}")
            raise

    def detect_faces(self, frame):
        """
        Phát hiện toàn bộ khuôn mặt xuất hiện trong ảnh/frame camera.
        Trả về danh sách các khuôn mặt kèm theo tọa độ land-marks.
        """
        if frame is None:
            return None
            
        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces

    def get_bbox(self, face):
        """
        Hàm bổ trợ lấy tọa độ Bounding Box từ dữ liệu khuôn mặt trả về của YuNet.
        Tiện lợi cho việc vẽ UI rectangle trong app.py hoặc đóng gói dữ liệu.
        """
        if face is None:
            return None
        # YuNet trả về mảng, trong đó 4 phần tử đầu tiên là: [x_min, y_min, width, height]
        return [int(face[0]), int(face[1]), int(face[2]), int(face[3])]