import cv2
import numpy as np
import logging
import config

logger = logging.getLogger("FaceEngine.AntiSpoofing")

class AntiSpoofing:
    def __init__(self, model_path = None, liveness_threshold=None):
        """
        Chuyên trách module chống giả mạo bằng MiniFASNetV2 (ONNX).
        Chạy độc lập ở tầng dưới.
        """
        self.model_path = model_path if model_path else config.MINIFASNET_WEIGHTS
        self.liveness_threshold = liveness_threshold if liveness_threshold is not None else config.LIVENESS_THRESHOLD

        try:
            self.liveness_net = cv2.dnn.readNetFromONNX(self.model_path)
            logger.info(f"Khởi tạo mô hình MiniFASNetV2 thành công (Threshold: {self.liveness_threshold}).")
        except Exception as e:
            logger.error(f"Không thể tải mô hình MiniFASNetV2 từ {self.model_path}. Chi tiết: {e}")
            raise

    def predict(self, frame, face_data, crop_scale=2.7) -> tuple[bool, float]:
        """
        Trích xuất vùng khuôn mặt theo tỷ lệ scale lớn (2.7) và chấm điểm thực thể sống.
        """
        try:
            if face_data is None:
                return False, 0.0

            h, w, _ = frame.shape
            x, y, box_w, box_h = face_data[0:4].astype(int)

            # Tính toán tâm khuôn mặt để mở rộng đều ra biên
            cx, cy = x + box_w // 2, y + box_h // 2
            
            max_side = max(box_w, box_h)
            new_size = int(max_side * crop_scale)

            # Đảm bảo không cắt lẹm ra ngoài kích thước khung hình gốc
            x1 = max(0, cx - new_size // 2)
            y1 = max(0, cy - new_size // 2)
            x2 = min(w, cx + new_size // 2)
            y2 = min(h, cy + new_size // 2)
            
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                return False, 0.0

            cropped_face = frame[y1:y2, x1:x2]
            
            # Chuẩn hóa Blob đầu vào cho MiniFASNetV2 (80x80)
            input_blob = cv2.dnn.blobFromImage(
                cropped_face, 
                scalefactor=1.0, 
                size=(80, 80), 
                mean=(0, 0, 0), 
                swapRB=True, 
                crop=False
            )

            self.liveness_net.setInput(input_blob)
            preds = self.liveness_net.forward()

            # Softmax chuyển đổi kết quả đầu ra
            exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
            prob = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
            
            real_prob = float(prob[0][1])  # Chỉ số index 1 đại diện cho Real

            if real_prob >= self.liveness_threshold:
                return True, real_prob
            return False, real_prob

        except Exception as e:
            logger.error(f"Lỗi hệ thống trong luồng check_liveness ngầm: {e}")
            return False, 0.0