import os

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG (SYSTEM PATHS)
# ==============================================================================
# Đường dẫn tuyệt đối đến thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thư mục chứa trọng số của các mô hình AI (Weights)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

YUNET_WEIGHTS = os.path.join(WEIGHTS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_WEIGHTS = os.path.join(WEIGHTS_DIR, "face_recognition_sface_2021dec.onnx")
MINIFASNET_WEIGHTS = os.path.join(WEIGHTS_DIR, "MiniFASNetV2.onnx")

# Thư mục gốc chứa dữ liệu lưu trữ cục bộ (Local Storage)
DATA_DIR = os.path.join(BASE_DIR, "data")

# Nơi lưu ảnh chân dung gốc lúc sinh viên đăng ký hệ thống
REGISTERED_FACES_DIR = os.path.join(DATA_DIR, "registered_faces")

# Nơi lưu ảnh chụp khoảnh khắc lúc sinh viên quét mặt điểm danh hàng ngày
ATTENDANCE_LOGS_DIR = os.path.join(DATA_DIR, "attendance_logs")


# ==============================================================================
# 2. CẤU HÌNH NGƯỠNG TỐI ƯU CHO AI (AI THRESHOLDS)
# ==============================================================================
# Ngưỡng phát hiện khuôn mặt của YuNet (Càng cao càng giảm bắt nhầm vật thể)
DET_THRESHOLD = 0.75

# Ngưỡng chống giả mạo của MiniFASNetV2 (Dưới ngưỡng này bị coi là Fake - ảnh/màn hình)
LIVENESS_THRESHOLD = 0.55

# Ngưỡng nhận diện chính xác danh tính của SFace (Tính theo Cosine Similarity)
RECOGNITION_THRESHOLD = 0.60


# ==============================================================================
# 3. CẤU HÌNH THIẾT BỊ ĐẦU CUỐI & THÔNG TIN ĐIỂM DANH (DEVICE & ATTENDANCE CONFIG)
# ==============================================================================
# Thông tin nhận diện phần cứng (Phục vụ đóng gói log và đồng bộ Web API sau này)
DEVICE_ID_CURRENT = "CAM_GATE_01"
LOCATION_CURRENT = "Cổng chính - Khu A"

# Khóa bí mật dùng để tạo chữ ký HMAC bảo mật gói tin khi đẩy lên Server Web
DEVICE_SECRET_KEY = "super_secret_hmac_key_for_gate_01"

# Chỉ số Camera kết nối (0 là Webcam mặc định, hoặc đường dẫn RTSP Stream của IP Cam)
CAMERA_SOURCE = 0 

# Interval bỏ qua khung hình (Xử lý 1 frame, bỏ qua 2 frame tiếp theo để giảm tải CPU)
SKIP_INTERVAL = 3


# ==============================================================================
# 4. LOGIC ĐIỂM DANH THÔNG MINH & GIỚI HẠN (ATTENDANCE LOGIC & COOLDOWN)
# ==============================================================================
# Thời gian chặn spam hiển thị UI cục bộ trên Camera (Tính bằng giây)
UI_COOLDOWN_SECONDS = 10

# Số giờ tối thiểu giữa lần quẹt mặt đầu tiên (IN) và lần tiếp theo để được tính là (OUT)
MIN_OUT_HOURS = 2.0

# Thời gian chặn spam ghi nhận log OUT liên tục vào Database (Tính bằng giây)
OUT_COOLDOWN_SECONDS = 300

# Số ngày lưu trữ ảnh chụp khoảnh khắc tối đa trước khi tự động xóa giải phóng ổ cứng
MAX_STORAGE_DAYS = 30


# ==============================================================================
# 5. CẤU HÌNH ĐỒ HỌA UI HỂN THỊ (UI GRAPHICS)
# ==============================================================================
FONT_PATH = "arial.ttf"
FONT_SIZE = 18

COLOR_SUCCESS = (0, 255, 0)     # Xanh lá: Điểm danh thành công
COLOR_WARNING = (0, 165, 255)   # Màu Cam: Ảnh mờ, lỗi chất lượng
COLOR_ALERT = (0, 0, 255)       # Đỏ: Giả mạo hoặc Người lạ
COLOR_DEFAULT = (255, 255, 255) # Trắng: Đang quét mặt