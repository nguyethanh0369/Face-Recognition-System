## 📂 Cấu Trúc Thư Mục Hệ Thống

```text
Face_Recognition_System/
│
├── core/                           # Logic cốt lõi xử lý các mô hình AI
│   ├── __init__.py
│   ├── anti_spoofing.py            # Quản lý và xử lý mô hình MiniFASNetV2 (Anti-Spoofing)
│   ├── detector.py                 # Quản lý và xử lý mô hình YuNet (Face Detection)
│   └── recognizer.py               # Quản lý và xử lý mô hình SFace (Face Recognition)
│
├── data/                           # Dữ liệu hình ảnh cục bộ (Local Storage)
│   ├── registered_faces/           # Ảnh khuôn mặt sinh viên đã đăng ký gốc trong hệ thống
│   └── attendance_logs/            # Ảnh chụp khoảnh khắc lúc sinh viên quẹt mặt điểm danh
|
├── database/                       # Quản lý dữ liệu và lưu trữ sinh viên
│   ├── __init__.py
│   ├── db_manager.py               # Thao tác với DB (Lưu log điểm danh, cập nhật trạng thái Real/Fake)
│   └── storage_manager.py          # Quản lý lưu trữ/xóa các tệp ảnh chụp từ camera để đối soát
│
├── weights/                        # Nơi lưu trữ tập trung các file mô hình ONNX
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── MiniFASNetV2.onnx
│
├── app.py                          # Giao diện ứng dụng (GUI/Web API) và điều hướng luồng xử lý chính
├── config.py                       # File cấu hình tập trung (Ngưỡng Threshold, Đường dẫn, Camera ID)