import os
import cv2
import numpy as np
import time
import datetime
import threading
import logging
import config
from .detector import FaceDetector

logger = logging.getLogger("FaceEngine.Recognizer")

class CooldownManager:
    def __init__(self, cooldown_seconds=15):
        self.cooldown_seconds = cooldown_seconds if cooldown_seconds is not None else config.UI_COOLDOWN_SECONDS
        self.cache = {}  
        self.lock = threading.Lock()

    def check_and_update(self, student_id, current_time=None):
        if current_time is None:
            current_time = time.time()
            
        with self.lock:
            expired_keys = [k for k, v in self.cache.items() if (current_time - v) >= self.cooldown_seconds]
            for k in expired_keys:
                del self.cache[k]
                
            last_time = self.cache.get(student_id)
            if last_time is not None and (current_time - last_time) < self.cooldown_seconds:
                return False  
                
            self.cache[student_id] = current_time
            return True  


class FaceRecognizer:
    def __init__(self, detector_model_path=None, recognizer_model_path=None, cooldown_seconds=None, det_threshold=None, recognition_threshold=None):
        """
        Bộ sinh trắc học chính: Chịu trách nhiệm Detect -> Quality Filter -> SFace 1:N Match.
        """
        # Khởi tạo các tham số dựa trên cấu hình tập trung config.py
        det_path = detector_model_path if detector_model_path is not None else config.YUNET_WEIGHTS
        rec_path = recognizer_model_path if recognizer_model_path is not None else config.SFACE_WEIGHTS
        d_thresh = det_threshold if det_threshold is not None else config.DET_THRESHOLD
        
        self.recognition_threshold = recognition_threshold if recognition_threshold is not None else config.RECOGNITION_THRESHOLD
        self.cooldown_seconds = cooldown_seconds if cooldown_seconds is not None else config.UI_COOLDOWN_SECONDS

        # Khởi tạo mô hình phát hiện khuôn mặt nội bộ (YuNet) từ package core
        self.detector_module = FaceDetector(det_path, d_thresh)
        self.detector = self.detector_module.detector 
        
        # Khởi tạo mô hình nhận diện khuôn mặt chuyên sâu (SFace)
        try:
            self.recognizer = cv2.FaceRecognizerSF.create(rec_path, "")
            logger.info("Khởi tạo mô hình SFace (Face Recognition) thành công.")
        except Exception as e:
            logger.error(f"Không thể khởi tạo mô hình SFace: {e}")
            raise

        self.lock = threading.Lock()
        self.student_ids = []
        self.student_names = []
        self.features_matrix = None 
        self.cooldown_manager = CooldownManager(self.cooldown_seconds)

    def get_bbox(self, face_data):
        return face_data[0:4].astype(int)

    def validate_face_quality(self, frame, face_data, is_registration=False):
        try:
            h, w, _ = frame.shape
            bbox = self.get_bbox(face_data)
            
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(w, bbox[0] + bbox[2]), min(h, bbox[1] + bbox[3])
            
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                return False, "Không thể xác định vùng khuôn mặt hợp lệ."
                
            cropped = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # 1. Kiểm tra độ sáng tổng thể (Brightness check)
            mean_brightness = float(np.mean(gray))
            if mean_brightness < 40: return False, f"Ảnh quá tối ({mean_brightness:.1f} < 40)"
            if mean_brightness > 240: return False, f"Ảnh quá sáng ({mean_brightness:.1f} > 240)"
            
            # 2. Kiểm tra độ sắc nét bằng toán tử Laplacian tránh ảnh nhòe khi di chuyển    
            min_sharpness = 40.0 if is_registration else 80.0
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur_score < min_sharpness:
                return False, f"Ảnh bị nhòe/mờ ({blur_score:.1f} < {min_sharpness})"

            # Trích xuất tọa độ mắt trái, mắt phải và mũi
            re_x, re_y = face_data[4], face_data[5]
            le_x, le_y = face_data[6], face_data[7]
            nose_x, nose_y = face_data[8], face_data[9]
            
            # 3. Kiểm tra góc xoay Roll của đầu (Nghiêng vai/cổ)           
            max_roll = 25.0 if is_registration else 15.0
            dy = le_y - re_y
            dx = le_x - re_x
            roll_angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
            if roll_angle > max_roll:
                return False, f"Đầu nghiêng quá mức ({roll_angle:.1f}° > {max_roll}°)"

            # 4. Kiểm tra góc quay Yaw (Ngoảnh mặt sang trái/phải thông qua tỷ lệ đối xứng mũi-mắt)
            min_sym = 0.35 if is_registration else 0.45
            dist_nose_re = abs(nose_x - re_x)
            dist_nose_le = abs(nose_x - le_x)
            max_dist = max(dist_nose_re, dist_nose_le)
            min_dist = min(dist_nose_re, dist_nose_le)
            
            if max_dist == 0: return False, "Lỗi xác định landmarks."
            symmetry_ratio = min_dist / max_dist
        
            if symmetry_ratio < min_sym:
                return False, f"Mặt quay nghiêng quá nhiều (độ đối xứng: {symmetry_ratio:.2f} < {min_sym})"
                
            return True, "Chất lượng ảnh đạt chuẩn."
        except Exception as e:
            return False, f"Lỗi trong quá trình kiểm tra chất lượng: {str(e)}"

    def set_known_faces(self, known_faces_dict):
        """
        Nạp dữ liệu từ DB, tính toán sẵn chuẩn hóa vector và nén thành ma trận RAM gọn nhẹ.
        """
        with self.lock:
            self.student_ids = []
            self.student_names = []
            features_list = []
            
            for student_id, data in known_faces_dict.items():
                self.student_ids.append(student_id)
                self.student_names.append(data["name"])
                features_list.append(data["feature"].flatten())
                
            if features_list:
                features_arr = np.array(features_list, dtype=np.float32)
                norms = np.linalg.norm(features_arr, axis=1, keepdims=True)
                self.features_matrix = features_arr / (norms + 1e-8)
                logger.info(f"Đã nạp ma trận đặc trưng {self.features_matrix.shape} cho tìm kiếm 1:N.")
            else:
                self.features_matrix = np.empty((0, 128), dtype=np.float32)

    def search_face_vectorized(self, query_feature, threshold=None):
        if threshold is None:
            threshold = self.recognition_threshold

        if self.features_matrix is None or len(self.student_ids) == 0:
            return None, "Unknown", 0.0
            
        with self.lock:
            q = query_feature.flatten()
            q_norm = q / (np.linalg.norm(q) + 1e-8)
            # Tích vô hướng ma trận tính toán độ tương đồng Cosine cực nhanh cho toàn bộ SV cùng lúc
            scores = np.dot(self.features_matrix, q_norm)
            
            max_idx = np.argmax(scores)
            max_score = float(scores[max_idx])
            
            if max_score >= threshold:
                return self.student_ids[max_idx], self.student_names[max_idx], max_score
            
            return None, "Unknown", max_score

    # HƯỚNG ĐI MỚI: Luồng xử lý loại bỏ chặn bước Liveness Check 
    def process_attendance_pipeline(self, frame, run_quality_check=False) -> tuple[str, float, str, np.ndarray]:
        """
        Luồng tối ưu cho điểm danh cực nhanh: Detect -> Quality Check -> SFace Search.
        Trả về thêm face_data để phục vụ lưu vùng ảnh crop (ROI) hoặc tái sử dụng ở luồng sau.
        Returns: (student_id, score, status_message, face_data)
        """
        try:
            faces = self.detect_faces(frame)

            if faces is None or len(faces) == 0:
                return None, 0.0, "NOT_FOUND: Không tìm thấy khuôn mặt nào.", None

            face_data = faces[0]
            
            # KIỂM TRA CHẤT LƯỢNG ẢNH
            if run_quality_check:
                is_valid, reason = self.validate_face_quality(frame, face_data)
                if not is_valid:
                    logger.warning(f"[QUALITY SHIELD] Từ chối do chất lượng ảnh: {reason}")
                    return None, 0.0, f"BAD_QUALITY: {reason}", face_data
            
            # TRÍCH XUẤT VÀ ĐỐI SÁNH 1:N 
            aligned_face = self.recognizer.alignCrop(frame, face_data)
            query_feature = self.recognizer.feature(aligned_face).astype(np.float32)
            
            student_id, name, score = self.search_face_vectorized(query_feature)
            
            if student_id is not None:
                # Đổi trạng thái từ VERIFIED sang PENDING_LIVENESS để biểu thị trạng thái chờ kiểm tra ngầm
                return student_id, score, "PENDING_LIVENESS", face_data
            return None, score, "UNKNOWN", face_data

        except Exception as e:
            logger.error(f"Lỗi trong luồng xử lý điểm danh: {e}")
            return None, 0.0, f"SYSTEM_ERROR: {str(e)}", None

    def extract_feature_from_image(self, image_source, is_registration=True):
        """
        Trích xuất vector khuôn mặt phục vụ đăng ký sinh viên mới.
        """
        try:
            if isinstance(image_source, bytes):
                nparr = np.frombuffer(image_source, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif isinstance(image_source, str):
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {image_source}")
                img = cv2.imread(image_source)
            elif isinstance(image_source, np.ndarray):
                img = image_source.copy()
            else:
                raise TypeError("Kiểu dữ liệu không được hỗ trợ.")
                
            if img is None:
                raise ValueError("Không thể giải mã hình ảnh.")
                
            faces = self.detector_module.detect_faces(img)
            if faces is None or len(faces) == 0:
                raise ValueError("Không tìm thấy khuôn mặt nào trong hình ảnh đăng ký.")
                
            face_data = faces[0]
            
            if is_registration:
                is_valid, reason = self.validate_face_quality(img, face_data, is_registration=True)
                if not is_valid:
                    raise ValueError(f"Chất lượng ảnh không đủ điều kiện đăng ký: {reason}")
                    
            aligned_face = self.recognizer.alignCrop(img, face_data)
            feature = self.recognizer.feature(aligned_face)
            return feature.astype(np.float32)
        except Exception as e:
            logger.error(f"Thao tác trích xuất vector khuôn mặt thất bại: {e}")
            raise

    def package_event(self, student_id, score, location=None, device_id=None, status="SUCCESS", reason=None):
        """
        Đóng gói sự kiện điểm danh định dạng JSON chuẩn. 
        Tự động lấy vị trí và tên camera mặc định từ config nếu không chỉ định cụ thể.
        """
        loc = location if location is not None else config.LOCATION_CURRENT
        dev = device_id if device_id is not None else config.DEVICE_ID_CURRENT

        return {
            "student_id": student_id if student_id else "Unknown",
            "score": float(score) if score is not None else 0.0,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
            "device_id": dev,
            "location": loc,
            "status": status,
            "reason": reason
        }
    
    def detect_faces(self, frame):
        """
        Phát hiện toàn bộ khuôn mặt xuất hiện trong khung hình thông qua mô hình YuNet.
        """
        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces