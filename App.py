import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from AI_logic import FaceEngine, FrameSkipper
from Database_manager import DatabaseManager

def draw_vietnamese_text(img, text, position, font_path="arial.ttf", font_size=18, color_rgb=(255, 255, 255)):
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
    # 1. Cấu hình vị trí đặt thiết bị điểm danh
    location_current = "Cổng chính - Khu A"
    device_id_current = "CAM_GATE_01"

    print("[HỆ THỐNG] Đang khởi tạo các mô hình AI...")
    
    # 2. Khởi tạo FaceEngine với các ngưỡng tối ưu tập trung (Tránh xung đột 3 model)
    # Thắt chặt YuNet (0.75), Giữ MiniFASNet trung tính (0.55), Đặt SFace nhận diện an toàn (0.60)
    engine = FaceEngine(
        weights_path="weights", 
        cooldown_seconds=10,        # Thời gian chặn spam điểm danh trùng lặp (10 giây)
        det_threshold=0.75, 
        liveness_threshold=0.55, 
        recognition_threshold=0.60
    )
    
    # Bỏ qua 2 khung hình, xử lý 1 khung hình để giảm tải CPU/GPU
    skipper = FrameSkipper(skip_interval=3) 
    
    # 3. Kết nối Cơ sở dữ liệu và tải danh sách sinh viên lên bộ nhớ RAM Matrix
    db = DatabaseManager()
    known_faces_dict = db.get_all_students() 
    # Cấu trúc DB cần trả về dạng: {student_id: {"name": student_name, "feature": np.ndarray}}
    
    engine.set_known_faces(known_faces_dict)
    print(f"[HỆ THỐNG] Khởi tạo thành công. Đã tải {len(known_faces_dict)} sinh viên lên RAM Cache.")
    

    # 4. Mở luồng Camera stream
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Sử dụng CAP_DSHOW giúp khởi động camera nhanh hơn trên Windows
    if not cap.isOpened():
        print("X LỖI: Không thể kết nối với Camera / Webcam.")
        return

    # Bộ đệm UI phẳng để đồng bộ trạng thái hiển thị mượt mà giữa các frame bị bỏ qua (skip frames)
    ui_state = {
        "bbox": None,
        "text_line1": "Đang quét...",
        "text_line2": "",
        "color_bgr": (255, 255, 255), # Trắng (Mặc định)
        "is_detected": False
    }

    print("\n>>> HỆ THỐNG ĐIỂM DANH FACEID SẴN SÀNG KHỞI CHẠY <<<\n")
    print("Bấm 'r' để ĐĂNG KÝ người mới")
    print("Bấm 'q' để THOÁT hệ thống\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Mất kết nối luồng hình ảnh từ Camera.")
            break
            
        frame = cv2.flip(frame, 1) # Lật gương camera để người đứng đối diện nhìn tự nhiên hơn
        display_frame = frame.copy()

        # 5. LUỒNG XỬ LÝ TRÍCH XUẤT AI CHÍNH (Chạy dựa trên bộ skip_interval)
        if skipper.should_process():
            # Chạy toàn bộ pipeline tích hợp: Detect -> Quality Check -> Liveness -> Search 1:N
            # Bật run_quality_check=True để kích hoạt lá chắn chống ảnh mờ/nhòe/nghiêng đầu sâu trong lõi AI
            student_id, score, status_msg = engine.process_attendance_pipeline(frame, run_quality_check=True)

            # Kiểm tra xem tầng Detector có tìm thấy khuôn mặt nào không
            h_f, w_f, _ = frame.shape
            engine.detector.setInputSize((w_f, h_f))
            _, faces = engine.detector.detect(frame)
            
            if faces is not None and len(faces) > 0:
                ui_state["is_detected"] = True
                ui_state["bbox"] = engine.get_bbox(faces[0])
                
                # PHÂN PHỐI TRẠNG THÁI UI VÀ GHI LOG DỰA VÀO STATUS_MESSAGE CỦA PIPELINE
                if status_msg == "VERIFIED":
                    # Trường hợp 1: Là sinh viên hợp lệ trong hệ thống
                    idx = engine.student_ids.index(student_id)
                    student_name = engine.student_names[idx]
                    
                    ui_state["text_line1"] = f"SV: {student_name} ({student_id})"
                    ui_state["text_line2"] = f"Match: {score*100:.1f}% | REAL"
                    ui_state["color_bgr"] = (0, 255, 0) # Xanh lá
                    
                    # Kiểm tra bộ đệm RAM chống spam điểm danh liên tục
                    if engine.cooldown_manager.check_and_update(student_id):
                        # Ghi nhận log trực tiếp vào cơ sở dữ liệu thông qua DatabaseManager
                        db_result = db.log_attendance(
                            student_id=student_id, 
                            location=location_current, 
                            score=score,
                            device_id=device_id_current
                        )
                        print(f"[ĐIỂM DANH THÀNH CÔNG] SV: {student_name} - Log ID: {db_result.get('log_id')}")

                elif "FAILED_LIVENESS" in status_msg:
                    # Trường hợp 2: Phát hiện hành vi giả mạo (Đưa ảnh điện thoại, ảnh in)
                    ui_state["text_line1"] = "CẢNH BÁO: GIẢ MẠO!"
                    ui_state["text_line2"] = f"Liveness Score: {score*100:.1f}% < Thresh"
                    ui_state["color_bgr"] = (0, 0, 255) # Đỏ rực
                    
                elif "BAD_QUALITY" in status_msg or "BAD_IMAGE" in status_msg:
                    # Trường hợp 3: Ảnh chụp không đạt chuẩn chất lượng (Mờ, nhòe, nghiêng quá mức)
                    reason_clean = status_msg.replace("BAD_QUALITY: ", "")
                    ui_state["text_line1"] = "Ảnh không đạt chuẩn"
                    ui_state["text_line2"] = reason_clean
                    ui_state["color_bgr"] = (0, 165, 255) # Màu Cam cảnh báo chất lượng

                else:
                    # Trường hợp 4: Khuôn mặt thật nhưng không nằm trong Database sinh viên (Người lạ)
                    ui_state["text_line1"] = "Người lạ (UNKNOWN)"
                    ui_state["text_line2"] = f"Khớp cao nhất: {score*100:.1f}% < {engine.recognition_threshold*100}%"
                    ui_state["color_bgr"] = (0, 0, 255) # Đỏ

            else:
                # Không tìm thấy bất kỳ khuôn mặt nào trên màn hình
                ui_state["is_detected"] = False
                ui_state["bbox"] = None
                ui_state["text_line1"] = "Đang quét..."
                ui_state["text_line2"] = ""
                ui_state["color_bgr"] = (255, 255, 255)

        # 6. ĐỒ HỌA UI LÊN KHUNG HÌNH (Render bounding box và Text đa dòng)
        if ui_state["is_detected"] and ui_state["bbox"] is not None:
            bbox = ui_state["bbox"]
            color_rgb = (ui_state["color_bgr"][2], ui_state["color_bgr"][1], ui_state["color_bgr"][0]) # Chuyển BGR sang RGB cho PIL
            
            # Vẽ hộp Bounding Box bao quanh mặt
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), ui_state["color_bgr"], 2)
            
            # Tính toán vị trí đặt Text thông minh (Ưu tiên đặt trên đỉnh hộp, nếu sát mép trên thì đẩy xuống đáy)
            text_y_start = bbox[1] - 50 if bbox[1] - 50 >= 0 else bbox[1] + bbox[3] + 10
            
            # Vẽ dòng trạng thái 1 (Tên/Cảnh báo) và dòng trạng thái 2 (Chỉ số % tương đồng/Lý do lỗi)
            display_frame = draw_vietnamese_text(display_frame, ui_state["text_line1"], (bbox[0], text_y_start), color_rgb=color_rgb)
            display_frame = draw_vietnamese_text(display_frame, ui_state["text_line2"], (bbox[0], text_y_start + 22), color_rgb=color_rgb)

        # 7. Hiển thị màn hình lên Window
        cv2.imshow("Hệ thống FaceID Điểm danh Sinh viên", display_frame)

        # 8. BẮT SỰ KIỆN PHÍM BẤM (KEYBOARD INTERACTION)
        key = cv2.waitKey(1) & 0xFF
        
        # Trường hợp nhấn nút 'r' để ĐĂNG KÝ NGƯỜI MỚI
        if key == ord('r'):
            print("\n[ĐĂNG KÝ] Đang dừng hình để thiết lập thông tin...")
            # Yêu cầu nhập thông tin từ Terminal
            new_id = input("Mã số sinh viên (MSSV): ").strip()
            new_name = input("Họ và tên: ").strip()
            
            if new_id and new_name:
                try:
                    # Gọi hàm trích xuất feature từ frame chụp hiện tại (Bật kiểm tra chất lượng gắt gao)
                    print("[ĐĂNG KÝ] Đang phân tích chất lượng khuôn mặt...")
                    new_feature = engine.extract_feature_from_image(frame, is_registration=True)
                    
                    # 1. Lưu dữ liệu mới xuống Database
                    db.save_new_student(student_id=new_id, name=new_name, feature=new_feature)
                    print(f"[DB SUCCESS] Đã lưu thành công SV {new_name} vào Database.")
                    
                    # 2. Đồng bộ hot-reload lại ma trận RAM để nhận diện được ngay lập tức
                    print("[ĐĂNG KÝ] Đang đồng bộ lại bộ nhớ RAM Cache...")
                    updated_faces_dict = db.get_all_students()
                    engine.set_known_faces(updated_faces_dict)
                    print("[HỆ THỐNG] Đồng bộ hoàn tất! Người mới đã có thể điểm danh ngay.\n")
                    
                except Exception as e:
                    print(f"[ĐĂNG KÝ THẤT BẠI]: {e}\n")
            else:
                print("Lỗi: Thông tin nhập vào không được để trống.\n")

        # Nhấn phím 'q' để tắt ứng dụng an toàn
        elif key == ord('q'):
            break

    # 9. Giải phóng tài nguyên hệ thống khi tắt ứng dụng
    cap.release()
    cv2.destroyAllWindows()
    print("[HỆ THỐNG] Ứng dụng đã đóng an toàn.")

if __name__ == "__main__":
    main()