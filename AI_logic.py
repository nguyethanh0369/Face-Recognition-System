import os
import cv2
import numpy as np
import time
import datetime
import threading
import logging

# Thiết lập log cho phần AI (AI monitoring setup)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FaceEngine")

# Tắt cảnh báo log OpenCV
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

class CooldownManager:
    def __init__(self, cooldown_seconds=15):
        """
        Quản lý bộ nhớ đệm chống spam điểm danh liên tiếp.
        Anti-spam mechanism cache with configurable cooldown in RAM.
        """
        self.cooldown_seconds = cooldown_seconds
        self.cache = {}  # Lưu dạng {student_id: timestamp}
        self.lock = threading.Lock()

    def check_and_update(self, student_id, current_time=None):
        """
        Kiểm tra ID xem có đang trong thời gian cooldown hay không.
        Nếu được phép, cập nhật timestamp mới.
        Check if student_id is in cooldown. If not, update timestamp.
        """
        if current_time is None:
            current_time = time.time()
            
        with self.lock:
            # Dọn dẹp cache quá hạn để tránh phình to bộ nhớ (Memory cleanup)
            expired_keys = [k for k, v in self.cache.items() if (current_time - v) >= self.cooldown_seconds]
            for k in expired_keys:
                del self.cache[k]
                
            last_time = self.cache.get(student_id)
            if last_time is not None and (current_time - last_time) < self.cooldown_seconds:
                return False  # Bị chặn do đang cooldown
                
            self.cache[student_id] = current_time
            return True  # Hợp lệ, cho phép ghi log


class FrameSkipper:
    def __init__(self, skip_interval=3):
        """
        Bộ đếm hỗ trợ bỏ qua frame để giảm tải CPU/GPU hệ thống.
        Frame skipping helper to reduce CPU/GPU workload.
        """
        self.skip_interval = max(1, skip_interval)
        self.frame_count = 0

    def should_process(self) -> bool:
        """
        Kiểm tra xem frame hiện tại có cần chạy AI hay không.
        Check if current frame should be processed.
        """
        self.frame_count += 1
        return (self.frame_count - 1) % self.skip_interval == 0

    def reset(self):
        """Reset bộ đếm frame."""
        self.frame_count = 0

class FaceEngine:
    def __init__(self, weights_path="weights", cooldown_seconds=15, det_threshold=0.75, liveness_threshold=0.55, recognition_threshold=0.60):
        """
        Khởi tạo bộ ba mô hình: YuNet (Detection), MiniFASNetV2 (Anti-Spoofing), và SFace (Recognition)
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(self.base_dir, weights_path)
        
        # Đường dẫn model ONNX
        detector_model = os.path.join(weights_dir, "face_detection_yunet_2023mar.onnx")
        recognizer_model = os.path.join(weights_dir, "face_recognition_sface_2021dec.onnx")
        liveness_model = os.path.join(weights_dir, "MiniFASNetV2.onnx")

        for model_path in [detector_model, recognizer_model, liveness_model]:
            if not os.path.exists(model_path):
                logger.error(f"X LỖI: Không tìm thấy file mô hình tại {model_path}")
                raise FileNotFoundError(f"X LỖI: Không tìm thấy file mô hình tại {model_path}")

        # --- QUY HOẠCH TOÀN BỘ NGƯỠNG ĐỂ KIỂM SOÁT TẬP TRUNG ---
        self.det_threshold = det_threshold                 # Ngưỡng phát hiện mặt (YuNet)
        self.liveness_threshold = liveness_threshold       # Ngưỡng thực thể sống (MiniFASNet)
        self.recognition_threshold = recognition_threshold # Ngưỡng khớp danh tính (SFace)

        # 1. Khởi tạo detector (YuNet)
        self.detector = cv2.FaceDetectorYN.create(detector_model, "", (320, 320), self.det_threshold, 0.3)
        
        # 2. Khởi tạo recognizer (SFace)
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_model, "")

        # 3. Khởi tạo bộ chống giả mạo bằng OpenCV DNN (MiniFASNetV2)
        self.liveness_net = cv2.dnn.readNetFromONNX(liveness_model)

        # 4. Quản lý đồng bộ và cache nhận diện nhanh (1:N Vectorized Cache)
        self.lock = threading.Lock()
        self.student_ids = []
        self.student_names = []
        self.features_matrix = None 
        self.cooldown_manager = CooldownManager(cooldown_seconds)

    # ==========================================
    # CƠ CHẾ KIỂM TRA CHỐNG GIẢ MẠO (Anti-Spoofing Logic)
    # ==========================================
    def check_liveness(self, frame, face_data, crop_scale=2.7) -> tuple[bool, float]:
        """
        Cắt ảnh khuôn mặt theo tỷ lệ 2.7 và đưa vào MiniFASNetV2 để phân tích thực thể sống.
        Tính toán an toàn tọa độ bọc biên, tránh lệch tâm, tối ưu hóa kênh màu.
        Trả về: (is_real: bool, real_probability: float)
        """
        try:
            h, w, _ = frame.shape
            # Giả định face_data tuân theo YuNet chuẩn: [x, y, width, height]
            x = int(face_data[0])
            y = int(face_data[1])
            box_w = int(face_data[2])
            box_h = int(face_data[3])

            # Tính toán tâm khuôn mặt
            cx, cy = x + box_w // 2, y + box_h // 2
            
            # Mở rộng bounding box theo tỷ lệ Crop Scale (2.7 cho MiniFASNetV2)
            max_side = max(box_w, box_h)
            new_size = int(max_side * crop_scale)

            # Xác định tọa độ vùng cắt mới
            x1 = max(0, cx - new_size // 2)
            y1 = max(0, cy - new_size // 2)
            x2 = min(w, cx + new_size // 2)
            y2 = min(h, cy + new_size // 2)
            
            # Kiểm tra kích thước hợp lệ
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                return False, 0.0

            # Cắt và resize về kích thước chuẩn của MiniFASNet (thường là 80x80)
            cropped_face = frame[y1:y2, x1:x2]
            input_blob = cv2.dnn.blobFromImage(cropped_face, 1.0, (80, 80), (0, 0, 0), swapRB=True, crop=False)

            # Đưa qua mạng mạng nơ-ron MiniFASNetV2
            self.liveness_net.setInput(input_blob)
            preds = self.liveness_net.forward()

            # Giả định đầu ra trả về mảng Softmax [Xác suất Fake, Xác suất Real]
            # Mẹo: Hàm Softmax đôi khi trả về logits, ta lấy chỉ số index hoặc exponentials
            exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
            prob = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
            
            real_prob = float(prob[0][1]) # Index 1 đại diện cho Thực thể sống (Real)

            if real_prob >= self.liveness_threshold:
                return True, real_prob
            return False, real_prob

        except Exception as e:
            logger.error(f"Lỗi hệ thống trong luồng check_liveness: {e}")
            return False, 0.0

    # ==========================================
    # LUỒNG XỬ LÝ CHÍNH (Processing Pipeline)
    # ==========================================
    def process_attendance_pipeline(self, frame, run_quality_check=False) -> tuple[str, float, str]:
        """
        Luồng phức hợp tối ưu cho API Điểm danh (Đã tích hợp Bộ lọc tích lũy chống Jittering): 
        Detect -> Liveness Check (Smoothing) -> Quality Check -> SFace Search.
        Trả về: (student_id, score, status_message)
        """
        try:
            h, w, _ = frame.shape
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)

            if faces is None or len(faces) == 0:
                return None, 0.0, "NOT_FOUND: Không tìm thấy khuôn mặt nào."

            face_data = faces[0]
            
            # BƯỚC 1: KIỂM TRA CHỐNG GIẢ MẠO (ANTI-SPOOFING)
            is_real, liveness_score = self.check_liveness(frame, face_data) # Sửa truyền frame lớn gốc để trích xuất chuẩn theo tỉ lệ 2.7
            if not is_real:
                return None, liveness_score, "FAILED_LIVENESS: Phát hiện ảnh chụp hoặc thiết bị giả mạo."
            
            # BƯỚC 2: KIỂM TRA CHẤT LƯỢNG ẢNH ĐỂ CHẶN CHIÊU TRÒ LẮC/DI CHUYỂN ẢNH
            if run_quality_check:
                is_valid, reason = self.validate_face_quality(frame, face_data)
                if not is_valid:
                    # Nếu ảnh bị nhòe do đối phương di chuyển điện thoại cố tình qua mắt AI, chặn luôn tại đây
                    logger.warning(f"[QUALITY SHIELD] Từ chối frame do chất lượng ảnh: {reason}")
                    return None, 0.0, f"BAD_QUALITY: {reason}"
            
            # Cắt vùng khuôn mặt (Face ROI) an toàn
            bbox = self.get_bbox(face_data)
            x, y, bw, bh = bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None, 0.0, "BAD_IMAGE: Vùng khuôn mặt không hợp lệ."
            
            
            # BƯỚC 3: TRÍCH XUẤT ĐỐI SÁNH VECTOR 1:N TRÊN RAM CACHE
            aligned_face = self.recognizer.alignCrop(frame, face_data)
            query_feature = self.recognizer.feature(aligned_face).astype(np.float32)
            
            student_id, name, score = self.search_face_vectorized(query_feature)
            
            if student_id is not None:
                return student_id, score, "VERIFIED"
            return None, score, "UNKNOWN"

        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong luồng xử lý điểm danh: {e}")
            return None, 0.0, f"SYSTEM_ERROR: {str(e)}"

    # ==========================================
    # CÁC PHƯƠNG THỨC XỬ LÝ KHUÔN MẶT CƠ BẢN (Base Face Processing)
    # ==========================================
    def get_feature(self, frame):
        """
        Phát hiện khuôn mặt và trích xuất vector đặc trưng.
        Trả về: (feature, face_coords) hoặc (None, None)
        """
        try:
            h, w, _ = frame.shape
            self.detector.setInputSize((w, h))
            
            # Phát hiện khuôn mặt
            _, faces = self.detector.detect(frame)
            
            if faces is not None and len(faces) > 0:
                # Lấy khuôn mặt có kích thước lớn nhất hoặc đầu tiên
                # Căn chỉnh và cắt khuôn mặt (Align and Crop)
                aligned_face = self.recognizer.alignCrop(frame, faces[0])
                # Trích xuất đặc trưng
                feature = self.recognizer.feature(aligned_face)
                return feature.astype(np.float32), faces[0]
            
            return None, None
        except Exception as e:
            logger.error(f"Lỗi trong get_feature: {e}")
            return None, None

    def compare(self, feature1, feature2):
        """
        So sánh độ tương đồng giữa 2 vector khuôn mặt bằng Cosine Similarity
        Giữ nguyên cơ chế cũ để đảm bảo tương thích ngược.
        Compare similarity of 2 vectors for backward compatibility.
        """
        try:
            score = self.recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
            return float(score)
        except Exception as e:
            logger.error(f"Lỗi trong so sánh compare: {e}")
            return 0.0

    def get_bbox(self, face_data):
        """
        Hỗ trợ lấy tọa độ khung bao (bounding box) từ dữ liệu khuôn mặt.
        Get bounding box coordinates from face data.
        """
        return face_data[0:4].astype(int)

    # ==========================================
    # 1:N VECTORIZED MATCHING (Numpy Optimizations)
    # ==========================================
    def set_known_faces(self, known_faces_dict):
        """
        Nạp danh sách sinh viên từ DB lên RAM dưới dạng ma trận NumPy.
        Convert student list to L2-normalized NumPy matrix for vectorized 1:N matching.
        known_faces_dict: {student_id: {"name": name, "feature": np.ndarray}}
        """
        with self.lock:
            self.student_ids = []
            self.student_names = []
            features_list = []
            
            for student_id, data in known_faces_dict.items():
                self.student_ids.append(student_id)
                self.student_names.append(data["name"])
                # Đảm bảo vector đặc trưng phẳng (flatten)
                features_list.append(data["feature"].flatten())
                
            if features_list:
                features_arr = np.array(features_list, dtype=np.float32)
                # Chuẩn hóa L2 cho ma trận để tối ưu tính toán Cosine (L2 normalization)
                norms = np.linalg.norm(features_arr, axis=1, keepdims=True)
                self.features_matrix = features_arr / (norms + 1e-8)
                logger.info(f"Đã nạp ma trận đặc trưng {self.features_matrix.shape} cho tìm kiếm 1:N.")
            else:
                self.features_matrix = np.empty((0, 128), dtype=np.float32)

    def search_face_vectorized(self, query_feature, threshold=None):
        """
        Thực hiện tìm kiếm 1:N vector chuẩn hóa tốc độ mili-giây.
        Perform fast 1:N cosine similarity search on RAM using NumPy.
        Returns: (student_id, name, score) hoặc (None, "Unknown", score)
        """
        if threshold is None:
            threshold = self.recognition_threshold # Sử dụng ngưỡng tập trung từ cấu hình Engine

        if self.features_matrix is None or len(self.student_ids) == 0:
            return None, "Unknown", 0.0
            
        with self.lock:
            # 1. Chuẩn hóa L2 vector truy vấn (L2 normalize query feature)
            q = query_feature.flatten()
            q_norm = q / (np.linalg.norm(q) + 1e-8)
            
            # 2. Tính tích vô hướng tương đồng Cosine cực nhanh (Vectorized dot product)
            scores = np.dot(self.features_matrix, q_norm)
            
            # 3. Lấy chỉ mục và điểm cao nhất
            max_idx = np.argmax(scores)
            max_score = float(scores[max_idx])
            
            if max_score >= threshold:
                return self.student_ids[max_idx], self.student_names[max_idx], max_score
            
            return None, "Unknown", max_score

    # ==========================================
    # THẨM ĐỊNH CHẤT LƯỢNG ẢNH (Image Quality Validation)
    # ==========================================
    def validate_face_quality(self, frame, face_data, is_registration=False):
        """
        Kiểm tra chất lượng khuôn mặt (Độ sáng, Độ sắc nét, Góc nghiêng khuôn mặt).
        Validate brightness, sharpness, and angles using face landmarks.
        Returns: (is_valid: bool, reason: str)
        """
        try:
            h, w, _ = frame.shape
            bbox = self.get_bbox(face_data)
            
            # Cắt vùng ảnh khuôn mặt
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(w, bbox[0] + bbox[2]), min(h, bbox[1] + bbox[3])
            
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                return False, "Không thể xác định vùng khuôn mặt hợp lệ."
                
            cropped = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # 1. Độ sáng (Brightness): Trung bình màu xám vùng mặt [50, 220]
            mean_brightness = float(np.mean(gray))
            if mean_brightness < 40:
                return False, f"Ảnh quá tối ({mean_brightness:.1f} < 40)"
            if mean_brightness > 240:
                return False, f"Ảnh quá sáng ({mean_brightness:.1f} > 240)"
                
            # 2. Độ nét (Sharpness): Laplacian variance > 80.0
            min_sharpness = 40.0 if is_registration else 80.0
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur_score < min_sharpness:
                return False, f"Ảnh bị nhòe/mờ ({blur_score:.1f} < {min_sharpness})"
                
            # 3. Góc mặt (Pose Check) sử dụng các điểm landmarks của YuNet
            # Landmark mappings:
            # Right eye: (face_data[4], face_data[5])
            # Left eye:  (face_data[6], face_data[7])
            # Nose tip:  (face_data[8], face_data[9])
            re_x, re_y = face_data[4], face_data[5]
            le_x, le_y = face_data[6], face_data[7]
            nose_x, nose_y = face_data[8], face_data[9]
            
            # Kiểm tra góc xoay đầu nghiêng (Roll tilt check)
            max_roll = 25.0 if is_registration else 15.0
            dy = le_y - re_y
            dx = le_x - re_x
            roll_angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
            if roll_angle > max_roll:
                return False, f"Đầu nghiêng quá mức ({roll_angle:.1f}° > {max_roll}°)"
                
            # Kiểm tra độ đối xứng ngang (Yaw check - xoay trái phải)
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

    # ==========================================
    # DỊCH VỤ ĐĂNG KÝ HÌNH ẢNH (Image Enrollment Service)
    # ==========================================
    def extract_feature_from_image(self, image_source, is_registration=True):
        """
        Trích xuất vector đặc trưng khuôn mặt từ file ảnh, stream bytes hoặc numpy array.
        Hỗ trợ đăng ký trực tiếp từ Web API.
        Extract features from path, bytes, or numpy array. Runs quality checks by default.
        """
        try:
            # 1. Đọc và giải mã dữ liệu ảnh đầu vào
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
                raise TypeError("Kiểu dữ liệu đầu vào không được hỗ trợ. Cần bytes, đường dẫn file, hoặc numpy array.")
                
            if img is None:
                raise ValueError("Không thể tải hoặc giải mã hình ảnh.")
                
            # 2. Chạy phát hiện khuôn mặt
            h, w, _ = img.shape
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(img)
            
            if faces is None or len(faces) == 0:
                raise ValueError("Không tìm thấy khuôn mặt nào trong hình ảnh đăng ký.")
                
            face_data = faces[0]
            
            # 3. Thẩm định chất lượng trước khi trích xuất vector đặc trưng
            if is_registration:
                is_valid, reason = self.validate_face_quality(img, face_data)
                if not is_valid:
                    raise ValueError(f"Chất lượng ảnh không đủ điều kiện đăng ký: {reason}")
                    
            # 4. Cắt khuôn mặt và trích xuất vector đặc trưng
            aligned_face = self.recognizer.alignCrop(img, face_data)
            feature = self.recognizer.feature(aligned_face)
            
            return feature.astype(np.float32)
        except Exception as e:
            logger.error(f"Thao tác trích xuất vector khuôn mặt thất bại: {e}")
            raise

    # ==========================================
    # ĐÓNG GÓI KẾT QUẢ SỰ KIỆN (Standardized Event Output)
    # ==========================================
    def package_event(self, student_id, score, location, device_id=None, status="SUCCESS", reason=None):
        """
        Đóng gói thông tin kết quả thành JSON/Dict chuẩn hóa gửi cho hệ thống Web API.
        Package recognition event details into a standardized structure for external APIs.
        """
        return {
            "student_id": student_id if student_id else "Unknown",
            "score": float(score) if score is not None else 0.0,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
            "device_id": device_id or "unknown_device",
            "location": location,
            "status": status,
            "reason": reason
        }