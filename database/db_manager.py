import sqlite3
import numpy as np
import os
import datetime
import hmac
import hashlib
import logging
import config

logger = logging.getLogger("DatabaseManager")

class DatabaseManager:
    def __init__(self, db_name="face_system.db"):
        """
        Khởi tạo kết nối SQLite, tạo các bảng và thực hiện tự động migration nếu cần.
        """
        # Sử dụng config.DATA_DIR làm nơi lưu trữ tệp CSDL tập trung thay vì để lung tung
        self.db_path = os.path.join(config.DATA_DIR, db_name)

        # Đảm bảo thư mục chứa DB (thư mục data/) đã được tự động tạo trước
        os.makedirs(config.DATA_DIR, exist_ok=True)
        self.init_db()

    def _get_connection(self):
        """Tạo kết nối mới đến database và bật khóa ngoại."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self):
        """Khởi tạo cơ sở dữ liệu và tự động cập nhật schema."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 1. Bảng lưu trữ sinh viên (Students table)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS students (
                        student_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        feature BLOB,
                        status TEXT DEFAULT 'Active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 2. Bảng quản lý thiết bị đầu cuối (Devices table)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS devices (
                        device_id TEXT PRIMARY KEY,
                        location_name TEXT NOT NULL,
                        secret_key TEXT NOT NULL,
                        status TEXT DEFAULT 'Active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 3. Bảng lịch sử điểm danh (Attendance table)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT,
                        location TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        score REAL,
                        log_type TEXT CHECK(log_type IN ('IN', 'OUT')),
                        device_id TEXT,
                        is_liveness INTEGER DEFAULT 1, -- 1: Real, 0: Fake (Cập nhật trạng thái chống giả mạo)
                        FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE,
                        FOREIGN KEY (device_id) REFERENCES devices (device_id) ON DELETE SET NULL
                    )
                ''')
                
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_student_time ON attendance (student_id, timestamp);') 
                # Tự động nạp thiết bị cấu hình hiện tại để tránh lỗi Khóa ngoại ---
                cursor.execute('''
                    INSERT OR IGNORE INTO devices (device_id, location_name, secret_key, status)
                    VALUES (?, ?, ?, 'Active')
                ''', (config.DEVICE_ID_CURRENT, config.LOCATION_CURRENT, config.DEVICE_SECRET_KEY))

                conn.commit()
                
                # Tự động thực hiện Migrations
                self._migrate_db(conn)
                logger.info("Cơ sở dữ liệu đã được khởi tạo và kiểm tra schema thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo cơ sở dữ liệu: {e}")
            raise

    def _migrate_db(self, conn):
        """Kiểm tra và cập nhật cấu trúc bảng cho các phiên bản cũ."""
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(students)")
        student_cols = [col[1] for col in cursor.fetchall()]
        if 'status' not in student_cols:
            cursor.execute("ALTER TABLE students ADD COLUMN status TEXT DEFAULT 'Active'")
            
        cursor.execute("PRAGMA table_info(attendance)")
        attendance_cols = [col[1] for col in cursor.fetchall()]
        if 'log_type' not in attendance_cols:
            cursor.execute("ALTER TABLE attendance ADD COLUMN log_type TEXT CHECK(log_type IN ('IN', 'OUT'))")
        if 'device_id' not in attendance_cols:
            cursor.execute("ALTER TABLE attendance ADD COLUMN device_id TEXT REFERENCES devices(device_id)")
        if 'is_liveness' not in attendance_cols:
            cursor.execute("ALTER TABLE attendance ADD COLUMN is_liveness INTEGER DEFAULT 1")
        conn.commit()

    # ==========================================
    # QUẢN LÝ SINH VIÊN (Student CRUD)
    # ==========================================
    def save_student(self, student_id, name, feature, status='Active'):
        """Lưu hoặc ghi đè (Replace) thông tin sinh viên."""
        try:
            with self._get_connection() as conn:
                feature_blob = feature.tobytes()
                conn.execute('''
                    INSERT OR REPLACE INTO students (student_id, name, feature, status)
                    VALUES (?, ?, ?, ?)
                ''', (student_id, name, feature_blob, status))
                conn.commit()
                logger.info(f"Đã lưu sinh viên thành công: {student_id} - {name}")
                return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu sinh viên {student_id}: {e}")
            return False

    def save_new_student(self, student_id, name, feature, status='Active'):
        """Alias tương thích ứng dụng cũ."""
        return self.save_student(student_id, name, feature, status)

    def check_student_exists(self, student_id):
        """Kiểm tra xem mã sinh viên đã tồn tại trong hệ thống chưa"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT name FROM students WHERE student_id = ?", (student_id,))
                row = cursor.fetchone()
                return row is not None  # Trả về True nếu đã tồn tại hồ sơ
        except Exception as e:
            logger.error(f"Lỗi kiểm tra sinh viên tồn tại: {e}")
            return False

    def update_student_face(self, student_id, feature):
        """Chỉ cập nhật vector khuôn mặt cho sinh viên đã có sẵn hồ sơ"""
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    UPDATE students 
                    SET feature = ? 
                    WHERE student_id = ?
                ''', (feature.tobytes(), student_id))
                conn.commit()
                logger.info(f"Đã cập nhật khuôn mặt thành công cho SV: {student_id}")
                return True
        except Exception as e:
            logger.error(f"Lỗi cập nhật khuôn mặt SV {student_id}: {e}")
            return False
        
    def update_student_info(self, student_id, name=None, feature=None):
        """Cập nhật Tên hoặc Vector khuôn mặt của sinh viên (Sửa)."""
        try:
            updates, params = [], []
            if name:
                updates.append("name = ?")
                params.append(name)
            if feature is not None:
                updates.append("feature = ?")
                params.append(feature.tobytes())
            
            if not updates: return False
            params.append(student_id)
            
            query = f"UPDATE students SET {', '.join(updates)} WHERE student_id = ?"
            with self._get_connection() as conn:
                cursor = conn.execute(query, tuple(params))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Lỗi cập nhật thông tin SV {student_id}: {e}")
            return False

    def update_student_status(self, student_id, status):
        """Kích hoạt (Active) hoặc Vô hiệu hóa (Inactive) sinh viên."""
        if status not in ('Active', 'Inactive'):
            raise ValueError("Status phải là 'Active' hoặc 'Inactive'")
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("UPDATE students SET status = ? WHERE student_id = ?", (status, student_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật trạng thái SV {student_id}: {e}")
            return False

    def delete_student(self, student_id):
        """Xóa hoàn toàn sinh viên khỏi hệ thống (Xóa)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
                conn.commit()
                logger.info(f"Đã xóa hoàn toàn sinh viên khỏi CSDL: {student_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Lỗi khi xóa sinh viên {student_id}: {e}")
            return False

    def get_student_feature(self, student_id):
        """Lấy vector đặc trưng của sinh viên đang hoạt động (Xác thực 1:1)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT feature, status FROM students WHERE student_id = ?", (student_id,))
                row = cursor.fetchone()
                if row:
                    feature_blob, status = row
                    if status != 'Active': return None
                    return np.frombuffer(feature_blob, dtype=np.float32).reshape(1, -1)
                return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy feature của SV {student_id}: {e}")
            return None

    def get_all_students(self, include_inactive=False):
        """Chỉ nạp lên RAM những sinh viên đang Active VÀ ĐÃ CÓ khuôn mặt (feature không NULL)"""
        try:
            with self._get_connection() as conn:
                query = "SELECT student_id, name, feature, status FROM students WHERE feature IS NOT NULL AND LENGTH(feature) > 0"
                if not include_inactive:
                    query += " AND status = 'Active'"
                    
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                all_faces = {}
                for row in rows:
                    student_id, name, feature_blob, status = row
                    feature = np.frombuffer(feature_blob, dtype=np.float32).reshape(1, -1)
                    all_faces[student_id] = {"name": name, "feature": feature, "status": status}
                return all_faces
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách sinh viên lên RAM Cache: {e}")
            return {}

    # ==========================================
    # QUẢN LÝ THIẾT BỊ & CHỮ KÝ BẢO MẬT (Device CRUD & HMAC)
    # ==========================================
    def register_device(self, device_id, location_name, secret_key, status='Active'):
        """Đăng ký thiết bị phần cứng mới."""
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO devices (device_id, location_name, secret_key, status)
                    VALUES (?, ?, ?, ?)
                ''', (device_id, location_name, secret_key, status))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Lỗi khi đăng ký thiết bị {device_id}: {e}")
            return False

    def update_device(self, device_id, location_name=None, secret_key=None, status=None):
        """Sửa thông tin cấu hình hoặc khóa bí mật của thiết bị."""
        try:
            updates, params = [], []
            if location_name: updates.append("location_name = ?"), params.append(location_name)
            if secret_key: updates.append("secret_key = ?"), params.append(secret_key)
            if status: updates.append("status = ?"), params.append(status)
            
            if not updates: return False
            params.append(device_id)
            
            query = f"UPDATE devices SET {', '.join(updates)} WHERE device_id = ?"
            with self._get_connection() as conn:
                cursor = conn.execute(query, tuple(params))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật thiết bị {device_id}: {e}")
            return False

    def delete_device(self, device_id):
        """Xóa thiết bị khỏi hệ thống."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM devices WHERE device_id = ?", (device_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Lỗi khi xóa thiết bị {device_id}: {e}")
            return False

    def get_device(self, device_id):
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT device_id, location_name, secret_key, status FROM devices WHERE device_id = ?", (device_id,))
                row = cursor.fetchone()
                if row:
                    return {"device_id": row[0], "location_name": row[1], "secret_key": row[2], "status": row[3]}
                return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin thiết bị {device_id}: {e}")
            return None

    def verify_device_signature(self, device_id, signature, message_timestamp, max_skew_seconds=300):
        """Xác thực bảo mật HMAC chống giả mạo gói tin và chống trùng lặp (Replay attack)."""
        try:
            device = self.get_device(device_id)
            if not device or device['status'] != 'Active': return False

            try:
                device_time = datetime.datetime.fromisoformat(message_timestamp.replace("Z", "+00:00"))
            except Exception: return False

            now_utc = datetime.datetime.now(datetime.timezone.utc)
            
            max_skew = getattr(config, "MAX_SKEW_SECONDS", 300)
            if abs((now_utc - device_time).total_seconds()) > max_skew: return False

            message = f"{device_id}:{message_timestamp}"
            expected_signature = hmac.new(
                device['secret_key'].encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(signature.lower(), expected_signature.lower())
        except Exception as e:
            logger.error(f"Lỗi trong quá trình xác thực thiết bị: {e}")
            return False

    # ==========================================
    # LOGIC ĐIỂM DANH TỰ ĐỘNG IN/OUT & LIVENESS
    # ==========================================
    def log_attendance(self, student_id, location, score, device_id=None, is_liveness=1, min_out_hours=2.0, out_cooldown_seconds=300):
        """Ghi nhận log điểm danh, tự động phân tích loại IN/OUT và lưu trạng thái Real/Fake."""
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        # Đọc dữ liệu trực tiếp từ file cấu hình tập trung
        min_out_hours = config.MIN_OUT_HOURS
        out_cooldown_seconds = config.OUT_COOLDOWN_SECONDS
        
        def make_result(status, log_type=None, reason=None, timestamp_str=None):
            ts = timestamp_str or now_utc.strftime("%Y-%m-%d %H:%M:%S")
            res = {
                "student_id": student_id, "location": location,
                "score": float(score) if score is not None else 0.0,
                "timestamp": ts, "log_type": log_type, "device_id": device_id, 
                "is_liveness": is_liveness, "status": status
            }
            if reason: res["reason"] = reason
            return res

        try:
            # 1. Kiểm tra giả mạo trước khi xử lý điểm danh (Cập nhật trạng thái Real/Fake)
            if is_liveness == 0:
                timestamp_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")
                with self._get_connection() as conn:
                    conn.execute('''
                        INSERT INTO attendance (student_id, location, score, log_type, device_id, is_liveness, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (student_id, location, score, 'IN', device_id, 0, timestamp_str))
                    conn.commit()
                logger.warning(f"CẢNH BÁO: Phát hiện khuôn mặt giả mạo (Fake) từ SV {student_id}")
                return make_result("LOGGED_FAKE", log_type="IN", reason="Fake Face Detected")

            # 2. Tính toán khoảng thời gian trong ngày (UTC)
            now_local = datetime.datetime.now().astimezone()
            start_utc = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(datetime.timezone.utc)
            end_utc = now_local.replace(hour=23, minute=59, second=59, microsecond=999999).astimezone(datetime.timezone.utc)
            
            start_utc_str = start_utc.strftime("%Y-%m-%d %H:%M:%S")
            end_utc_str = end_utc.strftime("%Y-%m-%d %H:%M:%S")

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, log_type FROM attendance 
                    WHERE student_id = ? AND is_liveness = 1 AND timestamp BETWEEN ? AND ? 
                    ORDER BY timestamp ASC
                ''', (student_id, start_utc_str, end_utc_str))
                
                rows = cursor.fetchall()
                log_type = 'IN'
                
                if rows:
                    def parse_db_time(ts_str):
                        try: return datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
                        except ValueError: return datetime.datetime.fromisoformat(ts_str.replace("Z", "")).replace(tzinfo=datetime.timezone.utc)

                    first_log_time = parse_db_time(rows[0][0])
                    last_log_time = parse_db_time(rows[-1][0])
                    last_log_type = rows[-1][1]
                    
                    if last_log_type == 'OUT':
                        cooldown_diff = (now_utc - last_log_time).total_seconds()
                        if cooldown_diff < out_cooldown_seconds:
                            return make_result("SKIPPED", log_type="OUT", reason="Spam OUT cooldown", timestamp_str=rows[-1][0])
                        log_type = 'OUT'
                    else: # 'IN'
                        hours_diff = (now_utc - first_log_time).total_seconds() / 3600.0
                        if hours_diff >= min_out_hours:
                            log_type = 'OUT'
                        else:
                            return make_result("SKIPPED", log_type="IN", reason="Inside minimum work period", timestamp_str=rows[-1][0])

                timestamp_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute('''
                    INSERT INTO attendance (student_id, location, score, log_type, device_id, is_liveness, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (student_id, location, score, log_type, device_id, 1, timestamp_str))
                conn.commit()
                
                return make_result("LOGGED", log_type=log_type, timestamp_str=timestamp_str)
        except Exception as e:
            logger.error(f"Lỗi ghi nhận điểm danh cho SV {student_id}: {e}")
            return make_result("ERROR", reason=str(e))