import sqlite3
import numpy as np
import os
import datetime
import hmac
import hashlib
import logging

# Thiết lập log để giám sát hệ thống (Logging setup for monitoring)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatabaseManager")

class DatabaseManager:
    def __init__(self, db_name="face_system.db"):
        """
        Khởi tạo kết nối SQLite, tạo các bảng và thực hiện tự động migration nếu cần.
        Initialize SQLite connection, create tables, and perform automatic migrations.
        """
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_name)
        self.init_db()

    def _get_connection(self):
        """
        Tạo kết nối mới đến database.
        Create a new connection to the database.
        """
        conn = sqlite3.connect(self.db_path)
        # Bật tính năng khóa ngoại (Foreign key support)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self):
        """
        Khởi tạo cơ sở dữ liệu: Tạo bảng và tự động cập nhật schema (migrations).
        Initialize database: Create tables and auto-update schema.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. Bảng lưu trữ sinh viên (Students table)
            # Thêm trường 'status' để quản lý kích hoạt/vô hiệu hóa
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    feature BLOB NOT NULL,
                    status TEXT DEFAULT 'Active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 2. Bảng quản lý thiết bị đầu cuối (Devices table)
            # Dùng để xác thực và phân quyền cho camera/thiết bị IoT
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
            # Thêm 'log_type' (IN/OUT) và 'device_id' để lưu thiết bị quét
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    location TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    score REAL,
                    log_type TEXT CHECK(log_type IN ('IN', 'OUT')),
                    device_id TEXT,
                    FOREIGN KEY (student_id) REFERENCES students (student_id),
                    FOREIGN KEY (device_id) REFERENCES devices (device_id)
                )
            ''')
            
            # Tạo index tối ưu tìm kiếm tốc độ cao (Create index for high-performance search)
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_student_time ON attendance (student_id, timestamp);')
            conn.commit()
            
            # 4. Tự động thực hiện Migrations nếu database đã tồn tại từ trước
            self._migrate_db(conn)
            
            conn.close()
            logger.info("Cơ sở dữ liệu đã được khởi tạo và kiểm tra schema thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo cơ sở dữ liệu: {e}")
            raise

    def _migrate_db(self, conn):
        """
        Kiểm tra và cập nhật cấu trúc bảng cho các phiên bản cũ.
        Check and migrate table schemas for backward compatibility.
        """
        cursor = conn.cursor()
        
        # Kiểm tra bảng students
        cursor.execute("PRAGMA table_info(students)")
        student_cols = [col[1] for col in cursor.fetchall()]
        if 'status' not in student_cols:
            logger.warning("Đang cập nhật schema: Thêm cột 'status' vào bảng 'students'...")
            cursor.execute("ALTER TABLE students ADD COLUMN status TEXT DEFAULT 'Active'")
            conn.commit()
            
        # Kiểm tra bảng attendance
        cursor.execute("PRAGMA table_info(attendance)")
        attendance_cols = [col[1] for col in cursor.fetchall()]
        if 'log_type' not in attendance_cols:
            logger.warning("Đang cập nhật schema: Thêm cột 'log_type' vào bảng 'attendance'...")
            cursor.execute("ALTER TABLE attendance ADD COLUMN log_type TEXT CHECK(log_type IN ('IN', 'OUT'))")
            conn.commit()
            
        if 'device_id' not in attendance_cols:
            logger.warning("Đang cập nhật schema: Thêm cột 'device_id' vào bảng 'attendance'...")
            cursor.execute("ALTER TABLE attendance ADD COLUMN device_id TEXT REFERENCES devices(device_id)")
            conn.commit()

    # ==========================================
    # QUẢN LÝ THÀNH VIÊN / SINH VIÊN (Student Management)
    # ==========================================
    def save_student(self, student_id, name, feature, status='Active'):
        """
        Lưu hoặc cập nhật thông tin sinh viên và vector đặc trưng vào DB.
        Save or replace student information and facial feature vector.
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Chuyển vector numpy array sang bytes để lưu vào kiểu BLOB
            feature_blob = feature.tobytes()
            
            cursor.execute('''
                INSERT OR REPLACE INTO students (student_id, name, feature, status)
                VALUES (?, ?, ?, ?)
            ''', (student_id, name, feature_blob, status))
            
            conn.commit()
            logger.info(f"Đã lưu sinh viên thành công: {student_id} - {name} (Trạng thái: {status})")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu sinh viên {student_id}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def save_new_student(self, student_id, name, feature, status='Active'):
        """
        Alias hoặc hàm bổ sung để khớp với code gọi trong app.py.
        Gọi hàm save_student đã có sẵn để tránh lặp code.
        """
        return self.save_student(student_id, name, feature, status)

    def update_student_status(self, student_id, status):
        """
        Kích hoạt hoặc vô hiệu hóa quyền điểm danh của sinh viên mà không xóa dữ liệu.
        Enable/disable student attendance rights without deleting their profile.
        """
        if status not in ('Active', 'Inactive'):
            raise ValueError("Status chỉ nhận giá trị 'Active' hoặc 'Inactive'")
            
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE students SET status = ? WHERE student_id = ?", (status, student_id))
            conn.commit()
            updated = cursor.rowcount > 0
            if updated:
                logger.info(f"Đã cập nhật trạng thái SV {student_id} thành {status}")
            else:
                logger.warning(f"Không tìm thấy SV {student_id} để cập nhật trạng thái.")
            return updated
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật trạng thái SV {student_id}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def get_student_feature(self, student_id):
        """
        Truy vấn vector đặc trưng của một sinh viên cụ thể (Hỗ trợ xác thực 1:1)
        Query the feature vector of a specific student.
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT feature, status FROM students WHERE student_id = ?", (student_id,))
            row = cursor.fetchone()
            if row:
                feature_blob, status = row
                if status != 'Active':
                    logger.warning(f"Sinh viên {student_id} đang bị vô hiệu hóa (Inactive).")
                    return None
                # Phục hồi numpy array từ bytes
                return np.frombuffer(feature_blob, dtype=np.float32).reshape(1, -1)
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy feature của SV {student_id}: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def get_all_students(self, include_inactive=False):
        """
        Lấy danh sách sinh viên để nạp vào RAM.
        Mặc định chỉ lấy sinh viên 'Active' để đưa vào nhận diện 1:N.
        Retrieve students list to load into RAM. Default fetches only 'Active' profiles.
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if include_inactive:
                cursor.execute("SELECT student_id, name, feature, status FROM students")
            else:
                cursor.execute("SELECT student_id, name, feature, status FROM students WHERE status = 'Active'")
                
            rows = cursor.fetchall()
            
            all_faces = {}
            for row in rows:
                student_id, name, feature_blob, status = row
                feature = np.frombuffer(feature_blob, dtype=np.float32).reshape(1, -1)
                all_faces[student_id] = {
                    "name": name, 
                    "feature": feature,
                    "status": status
                }
            return all_faces
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách sinh viên: {e}")
            return {}
        finally:
            if conn:
                conn.close()


    # ==========================================
    # QUẢN LÝ THIẾT BỊ & BẢO MẬT (Device & Security Management)
    # ==========================================
    def register_device(self, device_id, location_name, secret_key, status='Active'):
        """
        Đăng ký thiết bị camera hoặc thiết bị IoT phần cứng.
        Register a hardware device (camera/IoT terminal).
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO devices (device_id, location_name, secret_key, status)
                VALUES (?, ?, ?, ?)
            ''', (device_id, location_name, secret_key, status))
            conn.commit()
            logger.info(f"Đã đăng ký thiết bị: {device_id} tại {location_name}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi đăng ký thiết bị {device_id}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def get_device(self, device_id):
        """
        Lấy thông tin thiết bị từ DB.
        Get device details by ID.
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT device_id, location_name, secret_key, status FROM devices WHERE device_id = ?", (device_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "device_id": row[0],
                    "location_name": row[1],
                    "secret_key": row[2],
                    "status": row[3]
                }
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin thiết bị {device_id}: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def verify_device_signature(self, device_id, signature, message_timestamp, max_skew_seconds=300):
        """
        Xác thực chữ ký số HMAC-SHA256 gửi từ thiết bị phần cứng để chống giả mạo gói tin.
        Verify HMAC-SHA256 signature from terminal devices to prevent spoofing/replay attacks.
        """
        try:
            device = self.get_device(device_id)
            if not device or device['status'] != 'Active':
                logger.warning(f"Xác thực thất bại: Thiết bị {device_id} không tồn tại hoặc đã bị khóa.")
                return False

            # 1. Chống replay attack bằng cách kiểm tra lệch thời gian (skew time check)
            # message_timestamp định dạng ISO 8601 UTC (e.g. 2026-05-25T07:44:00Z)
            try:
                device_time = datetime.datetime.fromisoformat(message_timestamp.replace("Z", "+00:00"))
            except Exception as te:
                logger.warning(f"Lỗi định dạng timestamp của thiết bị: {te}")
                return False

            now_utc = datetime.datetime.now(datetime.timezone.utc)
            skew = abs((now_utc - device_time).total_seconds())
            if skew > max_skew_seconds:
                logger.warning(f"Xác thực thất bại: Lệch thời gian quá lớn ({skew:.1f}s > {max_skew_seconds}s)")
                return False

            # 2. Kiểm tra tính hợp lệ của chữ ký
            secret_key = device['secret_key']
            message = f"{device_id}:{message_timestamp}"
            
            expected_signature = hmac.new(
                secret_key.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # So sánh chữ ký an toàn chống Timing Attacks
            if hmac.compare_digest(signature.lower(), expected_signature.lower()):
                return True
            else:
                logger.warning(f"Xác thực thất bại: Chữ ký không trùng khớp đối với thiết bị {device_id}.")
                return False
        except Exception as e:
            logger.error(f"Lỗi trong quá trình xác thực thiết bị: {e}")
            return False


    # ==========================================
    # LOGIC ĐIỂM DANH TỰ ĐỘNG IN/OUT (Attendance & Automated IN/OUT Logic)
    # ==========================================
    def log_attendance(self, student_id, location, score, device_id=None, min_out_hours=2.0, out_cooldown_seconds=300):
        """
        Ghi lại lịch sử điểm danh với logic tự động tính toán IN/OUT:
        - Đồng bộ múi giờ bằng cách dùng Aware Datetimes (timezone.utc).
        - Tránh spam log 'OUT' liên tục bằng cách kiểm tra cooldown 5 phút từ bản ghi cuối cùng nếu là 'OUT'.
        - Tối ưu hóa hiệu năng truy vấn SQLite bằng index-friendly BETWEEN (Start of Day, End of Day).
        
        Log attendance with automatic IN/OUT decision logic:
        - Standardizes on timezone-aware datetimes.
        - Prevents duplicate 'OUT' log spam using a 5-minute cooldown.
        - Optimizes queries using range search over indexed column.
        """
        conn = None
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        # Cấu trúc kết quả đồng nhất để trả về cho API (Standardized output structure builder)
        def make_result(status, log_type=None, reason=None, timestamp_str=None):
            ts = timestamp_str or now_utc.strftime("%Y-%m-%d %H:%M:%S")
            res = {
                "student_id": student_id,
                "location": location,
                "score": float(score) if score is not None else 0.0,
                "timestamp": ts,
                "log_type": log_type,
                "device_id": device_id,
                "status": status
            }
            if reason:
                res["reason"] = reason
            return res

        try:
            # 1. Tính toán ranh giới Start/End của ngày hiện tại (Local Day) theo múi giờ UTC
            # Việc dùng BETWEEN thay cho hàm date() giúp tận dụng được index của cột timestamp
            now_local = datetime.datetime.now().astimezone()
            start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
            end_local = now_local.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            start_utc = start_local.astimezone(datetime.timezone.utc)
            end_utc = end_local.astimezone(datetime.timezone.utc)
            
            start_utc_str = start_utc.strftime("%Y-%m-%d %H:%M:%S")
            end_utc_str = end_utc.strftime("%Y-%m-%d %H:%M:%S")

            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 2. Truy vấn tối ưu bằng Index Range Scan
            cursor.execute('''
                SELECT timestamp, log_type 
                FROM attendance 
                WHERE student_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            ''', (student_id, start_utc_str, end_utc_str))
            
            rows = cursor.fetchall()
            
            log_type = 'IN' # Mặc định ban đầu
            
            if rows:
                # Helper phục hồi timezone-aware UTC datetime từ chuỗi CSDL
                def parse_db_time(ts_str):
                    try:
                        dt = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        dt = datetime.datetime.fromisoformat(ts_str.replace("Z", ""))
                    return dt.replace(tzinfo=datetime.timezone.utc)

                first_log_time = parse_db_time(rows[0][0])
                last_log_time = parse_db_time(rows[-1][0])
                last_log_type = rows[-1][1]
                
                if last_log_type == 'OUT':
                    # Kiểm tra cooldown chống ghi trùng lặp 'OUT'
                    cooldown_diff = (now_utc - last_log_time).total_seconds()
                    if cooldown_diff < out_cooldown_seconds:
                        reason = f"OUT log cooldown active ({cooldown_diff:.1f}s < {out_cooldown_seconds}s)"
                        logger.info(f"SKIPPED: SV {student_id} tại {location} - {reason}")
                        return make_result("SKIPPED", log_type="OUT", reason=reason, timestamp_str=rows[-1][0])
                    else:
                        log_type = 'OUT'
                else: # last_log_type == 'IN'
                    # Kiểm tra xem đã qua khoảng làm việc tối thiểu chưa để chuyển thành 'OUT'
                    hours_diff = (now_utc - first_log_time).total_seconds() / 3600.0
                    if hours_diff >= min_out_hours:
                        log_type = 'OUT'
                    else:
                        reason = f"Scan inside active work period ({hours_diff:.2f}h since IN)"
                        logger.info(f"SKIPPED: SV {student_id} tại {location} - {reason}")
                        return make_result("SKIPPED", log_type="IN", reason=reason, timestamp_str=rows[-1][0])

            # 3. Ghi bản ghi điểm danh mới đồng bộ giờ UTC
            timestamp_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('''
                INSERT INTO attendance (student_id, location, score, log_type, device_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (student_id, location, score, log_type, device_id, timestamp_str))
            
            conn.commit()
            logger.info(f"LOGGED: SV {student_id} điểm danh {log_type} tại {location} (Score: {score:.2f})")
            
            return make_result("LOGGED", log_type=log_type, timestamp_str=timestamp_str)
            
        except Exception as e:
            logger.error(f"ERROR: Lỗi khi ghi nhận điểm danh cho SV {student_id}: {e}")
            if conn:
                conn.rollback()
            return make_result("ERROR", reason=str(e))
        finally:
            if conn:
                conn.close()