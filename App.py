import cv2
import numpy as np
import os
import time
import config
from PIL import ImageFont, ImageDraw, Image

# Import cấu hình tập trung và các module chuyên trách
from core.detector import FaceDetector          
from core.recognizer import FaceRecognizer      
from core.anti_spoofing import AntiSpoofing     
from database.db_manager import DatabaseManager
from database.storage_manager import StorageManager

def draw_vietnamese_text(img, text, position, font_path=config.FONT_PATH, font_size=config.FONT_SIZE, color_rgb=(255, 255, 255)):
    """Hỗ trợ vẽ chữ tiếng Việt có dấu lên khung hình OpenCV bằng thư viện PIL"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    # 1. Cấu hình vị trí đặt thiết bị điểm danh (Thuận tiện để đóng gói thành biến môi trường khi lên Web)
    location_current = config.LOCATION_CURRENT
    device_id_current = config.DEVICE_ID_CURRENT

    print("[HỆ THỐNG] Đang khởi tạo các mô hình AI độc lập từ core/...")
    
    # 2. Khởi tạo các mô hình AI tách biệt từ package core/
    detector = FaceDetector(config.YUNET_WEIGHTS, config.DET_THRESHOLD)
    recognizer = FaceRecognizer()  # Khởi tạo không tham số để tự nạp config chuẩn bên trong core
    anti_spoofing = AntiSpoofing() # Tự nạp config chuẩn cho MiniFASNet

    # 3. Kết nối Cơ sở dữ liệu & Trình quản lý lưu trữ hình ảnh
    db = DatabaseManager()
    storage = StorageManager()
    
    # Tải danh sách sinh viên hợp lệ lên bộ nhớ RAM Cache để nhận diện 1:N tốc độ cao
    known_faces_dict = db.get_all_students(include_inactive=False) 
    recognizer.set_known_faces(known_faces_dict)
    print(f"[HỆ THỐNG] Khởi tạo thành công. Đã tải {len(known_faces_dict)} sinh viên Active lên RAM Cache.")

    # 4. Mở luồng Camera stream
    cap = cv2.VideoCapture(config.CAMERA_SOURCE, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("X LỖI: Không thể kết nối với Camera / Webcam.")
        return

    skip_interval = config.SKIP_INTERVAL        # Số frame sẽ bỏ qua để giảm tải CPU
    cooldown_seconds = config.UI_COOLDOWN_SECONDS # Thời gian đóng băng UI tránh spam hiển thị

    # Bộ quản lý chống spam (Cooldown) cục bộ cho luồng camera
    cooldown_dict = {} # {student_id: timestamp}
    frame_count = 0    # Bộ đếm số khung hình đã đi qua

    # Bộ đệm UI phẳng để đồng bộ trạng thái hiển thị mượt mà giữa các frame bị bỏ qua
    ui_state = {
        "bbox": None,
        "text_line1": "Đang quét...",
        "text_line2": "",
        "color_bgr": (255, 255, 255), 
        "is_detected": False
    }

    print("\n>>> HỆ THỐNG ĐIỂM DANH FACEID SẴN SÀNG KHỞI CHẠY <<<\n")
    print("Bấm 'r' để ĐĂNG KÝ người mới trực tiếp")
    print("Bấm 'q' để THOÁT hệ thống an toàn\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Mất kết nối luồng hình ảnh từ Camera.")
            break
            
        frame = cv2.flip(frame, 1) # Lật gương camera
        display_frame = frame.copy()

        # 5. LUỒNG XỬ LÝ AI PHÂN TÁCH (Xử lý theo chu kỳ Skip Frame)
        if frame_count % config.SKIP_INTERVAL == 0:
            # Bước 5.1: Phát hiện khuôn mặt (YuNet)
            student_id, score, status_msg, face_data = recognizer.process_attendance_pipeline(frame, run_quality_check=True)
            
            if face_data is not None:
                ui_state["is_detected"] = True
                ui_state["bbox"] = recognizer.get_bbox(face_data)
                
                # PHÂN PHỐI TRẠNG THÁI UI VÀ GHI LOG DỰA VÀO STATUS_MESSAGE CỦA PIPELINE
                if status_msg == "PENDING_LIVENESS":
                    # Khuôn mặt đã khớp 1:N thành công -> Tiếp tục kích hoạt kiểm tra thực thể ngầm (Liveness)
                    is_liveness, liveness_score = anti_spoofing.predict(frame, face_data)
                    
                    if not is_liveness:
                        # TRƯỜNG HỢP GIẢ MẠO (Ảnh chụp qua điện thoại, màn hình, ảnh in giấy)
                        ui_state["text_line1"] = "CẢNH BÁO: GIẢ MẠO!"
                        ui_state["text_line2"] = f"Liveness Score: {liveness_score*100:.1f}% < {config.LIVENESS_THRESHOLD*100}%"
                        ui_state["color_bgr"] = config.COLOR_ALERT 
                        
                        # Hệ thống tự động lưu trữ ảnh FAKE đối soát và đẩy Log cảnh báo vào DB
                        storage.save_image("UNKNOWN_FAKE", frame, is_liveness=0)
                        db.log_attendance("UNKNOWN", location_current, score=liveness_score, device_id=device_id_current, is_liveness=0)
                    else:
                        # TRƯỜNG HỢP SINH VIÊN HỢP LỆ VÀ LÀ NGƯỜI THẬT
                        try:
                            idx = recognizer.student_ids.index(student_id)
                            student_name = recognizer.student_names[idx]
                        except (ValueError, IndexError):
                            student_name = "Sinh viên"
                        
                        ui_state["text_line1"] = f"SV: {student_name} ({student_id})"
                        ui_state["text_line2"] = f"Match: {score*100:.1f}% | REAL"
                        ui_state["color_bgr"] = config.COLOR_SUCCESS 
                        
                        # Gọi bộ quản lý Cooldown tích hợp sẵn bên trong CooldownManager để chống spam điểm danh liên tục
                        if recognizer.cooldown_manager.check_and_update(student_id):
                            # Lưu ảnh khoảnh khắc quẹt mặt thực tế để Admin đối soát trực quan
                            storage.save_image(student_id, frame, is_liveness=1)
                            
                            # Ghi nhận log điểm danh thông minh (Tự động tính toán luồng logic IN/OUT) vào CSDL
                            db_result = db.log_attendance(
                                student_id=student_id, 
                                location=location_current, 
                                score=score,
                                device_id=device_id_current,
                                min_out_hours=config.MIN_OUT_HOURS,
                                out_cooldown_seconds=config.OUT_COOLDOWN_SECONDS
                            )
                            print(f"[ĐIỂM DANH] {db_result['status']} -> SV: {student_name} ({student_id}) | Log Type: {db_result.get('log_type')}")
                
                elif "BAD_QUALITY" in status_msg:
                    # TRƯỜNG HỢP KHÔNG ĐẠT CHUẨN CHẤT LƯỢNG (Ảnh mờ, nghiêng quá mức, tối quá)
                    reason_clean = status_msg.replace("BAD_QUALITY: ", "")
                    ui_state["text_line1"] = "Ảnh không đạt chuẩn"
                    ui_state["text_line2"] = reason_clean
                    ui_state["color_bgr"] = config.COLOR_WARNING 

                else:
                    # TRƯỜNG HỢP KHUÔN MẶT THẬT NHƯNG KHÔNG NẰM TRONG CSDL (Người lạ)
                    ui_state["text_line1"] = "Người lạ (UNKNOWN)"
                    ui_state["text_line2"] = f"Khớp cao nhất: {score*100:.1f}% < {config.RECOGNITION_THRESHOLD*100}%"
                    ui_state["color_bgr"] = config.COLOR_ALERT 
            else:
                # Không phát hiện thấy bất kỳ khuôn mặt nào trên màn hình
                ui_state["is_detected"] = False
                ui_state["bbox"] = None
                ui_state["text_line1"] = "Đang quét..."
                ui_state["text_line2"] = ""
                ui_state["color_bgr"] = config.COLOR_DEFAULT

        frame_count += 1

        # 6. VẼ ĐỒ HỌA UI ĐỒNG BỘ LÊN KHUNG HÌNH (Render UI)
        if ui_state["is_detected"] and ui_state["bbox"] is not None:
            bbox = ui_state["bbox"]
            color_rgb = (ui_state["color_bgr"][2], ui_state["color_bgr"][1], ui_state["color_bgr"][0])
            
            # Vẽ Bounding Box bao quanh khuôn mặt
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), ui_state["color_bgr"], 2)
            
            # Tính toán vị trí đặt chữ thông minh
            text_y_start = bbox[1] - 50 if bbox[1] - 50 >= 0 else bbox[1] + bbox[3] + 10
            display_frame = draw_vietnamese_text(display_frame, ui_state["text_line1"], (bbox[0], text_y_start), color_rgb=color_rgb)
            display_frame = draw_vietnamese_text(display_frame, ui_state["text_line2"], (bbox[0], text_y_start + 22), color_rgb=color_rgb)

        # Show màn hình OpenCV
        cv2.imshow("Hệ thống FaceID Điểm danh Sinh viên", display_frame)

        # 7. TƯƠNG TÁC PHÍM BẤM TERMINAL (Hỗ trợ quản trị viên quản lý nhanh)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            print("\n[ĐĂNG KÝ / CẬP NHẬT] Hệ thống tạm dừng luồng quét để thiết lập...")
            new_id = input("Nhập Mã số sinh viên (MSSV): ").strip()
            
            if not new_id:
                print("[THẤT BẠI]: MSSV không được để trống.\n")
                continue
            
            # Kiểm tra xem sinh viên này đã có hồ sơ trong DB chưa
            is_existing = db.check_student_exists(new_id)
            new_name = ""
            
            if is_existing:
                # Trường hợp 1: Sinh viên đã có tên sẵn trong danh sách nền
                with db._get_connection() as conn:
                    new_name = conn.execute("SELECT name FROM students WHERE student_id = ?", (new_id,)).fetchone()[0]
                print(f"-> Tìm thấy hồ sơ: SV [ {new_name} ]. Tiến hành chụp ảnh bổ sung khuôn mặt.")
            else:
                # Trường hợp 2: Sinh viên mới hoàn toàn, yêu cầu điền tên
                print("-> MSSV mới hoàn toàn. Vui lòng tạo hồ sơ mới.")
                new_name = input("Nhập Họ và tên sinh viên: ").strip()
                if not new_name:
                    print("[THẤT BẠI]: Tên sinh viên mới không được để trống.\n")
                    continue

            # Tiến hành trích xuất khuôn mặt từ camera ngay tại frame hiện tại
            faces_reg = detector.detect_faces(frame)
            if faces_reg is not None and len(faces_reg) > 0:
                try:
                    # Trích xuất vector đặc trưng kèm bộ lọc chất lượng ảnh
                    new_feature = recognizer.extract_feature_from_image(frame, is_registration=True)
                    
                    if is_existing:
                        # Chỉ cập nhật khuôn mặt vào hồ sơ có sẵn
                        db.update_student_face(student_id=new_id, feature=new_feature)
                    else:
                        # Tạo mới hoàn toàn cả thông tin và khuôn mặt
                        db.save_student(student_id=new_id, name=new_name, feature=new_feature, status='Active')
                    
                    # Lưu ảnh chân dung gốc làm tư liệu đối soát vật lý
                    storage.save_registered_face(new_id, frame)
                    
                    # HOT-RELOAD: Nạp lại ma trận RAM Cache ngay lập tức
                    updated_faces_dict = db.get_all_students(include_inactive=False)
                    recognizer.set_known_faces(updated_faces_dict)
                    
                    print(f"[THÀNH CÔNG] Đã đồng bộ dữ liệu cho SV {new_name} ({new_id}). Có thể điểm danh ngay!\n")
                except Exception as e:
                    print(f"[LỖI TRÍCH XUẤT]: Chất lượng khuôn mặt không đạt chuẩn. Chi tiết: {e}\n")
            else:
                print("[THẤT BẠI]: Không tìm thấy khuôn mặt nào trước camera để đăng ký.\n")

        elif key == ord('q'):
            break

    # 8. Giải phóng tài nguyên an toàn khi thoát ứng dụng
    cap.release()
    cv2.destroyAllWindows()
    print("[HỆ THỐNG] Ứng dụng đã đóng an toàn.")

if __name__ == "__main__":
    main()