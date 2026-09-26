r"""Ứng dụng desktop nhận diện khuôn mặt và phân tích trạng thái tài xế.

Chạy:
    .\.venv\Scripts\python.exe desktop_test_app.py

Yêu cầu tệp trọng số:
    - Mô hình phân loại: driver_drowsiness_cnn.keras và class_names.json
    - Mô hình phát hiện mặt: face_detection_yunet_2023mar.onnx
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import cv2
    import numpy as np
    import tensorflow as tf
except ModuleNotFoundError as exc:
    missing_package = exc.name or "thư viện cần thiết"
    raise SystemExit(
        f"Thiếu {missing_package} trong Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        "Hãy cài đặt đầy đủ opencv-python, numpy và tensorflow trong môi trường ảo."
    ) from exc


APP_DIR = Path(__file__).resolve().parent
WORKING_DIR = Path.cwd()

MODEL_CLASSIFIER_CANDIDATES = (
    APP_DIR / "driver_drowsiness_cnn.keras",
    APP_DIR / "best_finetuned_model.keras",
    APP_DIR / "driver_drowsiness_cnn.h5",
    WORKING_DIR / "driver_drowsiness_cnn.keras",
)

MODEL_YUNET_CANDIDATES = (
    APP_DIR / "face_detection_yunet_2023mar.onnx",
    WORKING_DIR / "face_detection_yunet_2023mar.onnx",
)

METADATA_FILENAME = "class_names.json"

DISPLAY_NAMES = {
    "alert": "Tỉnh táo",
    "drowsy": "Buồn ngủ / mắt nhắm",
    "yawning": "Đang ngáp",
}
STATUS_MESSAGES = {
    "alert": "TỈNH TÁO",
    "drowsy": "CẢNH BÁO: BUỒN NGỦ / MẮT NHẮM",
    "yawning": "CẢNH BÁO: ĐANG NGÁP",
}
STATUS_COLORS = {
    "alert": "#17823b",
    "drowsy": "#c62828",
    "yawning": "#d97706",
    "neutral": "#6b7280",
}

CONFIDENCE_THRESHOLD = 0.55
SMOOTHING_FRAMES = 5
PREDICTION_INTERVAL_SECONDS = 0.15
CAMERA_REFRESH_MS = 30
PREVIEW_WIDTH = 820
PREVIEW_HEIGHT = 560


@dataclass
class DetectedFace:
    x: int
    y: int
    w: int
    h: int
    confidence: float

    @property
    def area(self) -> int:
        return self.w * self.h

    def get_padded_box(
        self, frame_w: int, frame_h: int, pad_ratio_x: float = 0.15, pad_ratio_y: float = 0.20
    ) -> tuple[int, int, int, int]:
        # Bổ sung lề dọc lớn hơn lề ngang nhằm bao quát toàn bộ vùng cằm khi há miệng (ngáp)
        # và vùng trán tránh trường hợp mô hình mất ngữ cảnh biểu cảm khuôn mặt
        pad_x = int(self.w * pad_ratio_x)
        pad_y = int(self.h * pad_ratio_y)

        x1 = max(0, self.x - pad_x)
        y1 = max(0, self.y - pad_y)
        x2 = min(frame_w, self.x + self.w + pad_x)
        y2 = min(frame_h, self.y + self.h + pad_y)
        return x1, y1, x2, y2


class YuNetFaceDetector:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 1000,
    ):
        self._current_size = (320, 320)
        # Sử dụng backend mặc định của OpenCV CPU để đạt tốc độ xử lý trên 100 FPS mà không tiêu tốn GPU
        self._detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=self._current_size,
            score_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

    def detect(self, bgr_frame: np.ndarray) -> list[DetectedFace]:
        height, width = bgr_frame.shape[:2]

        # Tránh khởi tạo lại cấu trúc mạng khi tỷ lệ khung hình thay đổi, chỉ tái định hình input layer
        if self._current_size != (width, height):
            self._current_size = (width, height)
            self._detector.setInputSize(self._current_size)

        retval, raw_faces = self._detector.detect(bgr_frame)
        if raw_faces is None or retval == 0:
            return []

        results: list[DetectedFace] = []
        for face_data in raw_faces:
            raw_x, raw_y, raw_w, raw_h = face_data[0:4]
            confidence = float(face_data[14])

            # Chặn biên chặt chẽ nhằm triệt tiêu lỗi out-of-bounds ma trận khi đối tượng di chuyển sát cạnh camera
            x = max(0, int(raw_x))
            y = max(0, int(raw_y))
            w = min(width - x, max(0, int(raw_w)))
            h = min(height - y, max(0, int(raw_h)))

            if w > 0 and h > 0:
                results.append(DetectedFace(x=x, y=y, w=w, h=h, confidence=confidence))

        return results


def read_image_unicode(path: Path) -> np.ndarray | None:
    # Tránh lỗi đường dẫn Windows chứa khoảng trắng hoặc ký tự tiếng Việt có dấu khi dùng cv2.imread
    try:
        encoded_path = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(encoded_path, cv2.IMREAD_COLOR)
    except Exception:
        return None


def fit_inside(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    scale = min(width / frame_width, height / frame_height)
    new_width = max(1, int(frame_width * scale))
    new_height = max(1, int(frame_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def bgr_to_tk_image(frame: np.ndarray) -> tk.PhotoImage:
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Không thể mã hóa khung hình để hiển thị lên GUI.")
    png_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return tk.PhotoImage(data=png_base64)


class DriverStateApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hệ thống phát hiện ngủ gật tích hợp Face Detection")
        self.root.minsize(1160, 720)

        self.classifier_model: tf.keras.Model | None = None
        self.face_detector: YuNetFaceDetector | None = None
        self.class_names: list[str] = []
        self.input_size: tuple[int, int] = (224, 224)

        self.camera: cv2.VideoCapture | None = None
        self.camera_running = False
        self.camera_after_id: str | None = None
        self.last_prediction_time = 0.0
        self.last_probabilities: np.ndarray | None = None
        self.prediction_history: deque = deque(maxlen=SMOOTHING_FRAMES)
        self.last_primary_face: DetectedFace | None = None

        self.preview_photo: tk.PhotoImage | None = None
        self.class_rows: list[tuple[tk.StringVar, tk.DoubleVar]] = []

        self.mode_var = tk.StringVar(value="Sẵn sàng")
        self.model_var = tk.StringVar(value="Đang nạp mô hình...")
        self.detector_var = tk.StringVar(value="Đang kiểm tra detector...")
        self.status_var = tk.StringVar(value="CHƯA CÓ DỮ LIỆU")
        self.confidence_var = tk.StringVar(value="Độ tin cậy: --")
        self.camera_index_var = tk.IntVar(value=0)

        self._build_interface()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._initialize_models)

    def _build_interface(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Class.TLabel", font=("Segoe UI", 10))

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Giám sát trạng thái tài xế (Face Tracking + Trạng thái mắt/miệng)",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="Pipeline xử lý: Định vị ROI khuôn mặt qua YuNet -> Chuẩn hóa bounding box -> Dự đoán trạng thái.",
        ).pack(anchor="w", pady=(2, 10))

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 10))

        ttk.Button(controls, text="Chọn Classifier Model", command=self.choose_classifier_model).pack(side="left")
        ttk.Button(controls, text="Chọn Ảnh Kiểm Tra", command=self.choose_image).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Bật Camera", command=self.start_camera).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Dừng Camera", command=self.stop_camera).pack(side="left", padx=(8, 0))

        ttk.Label(controls, text="Chỉ số Camera:").pack(side="left", padx=(16, 4))
        ttk.Spinbox(controls, from_=0, to=5, width=4, textvariable=self.camera_index_var).pack(side="left")

        ttk.Button(controls, text="Thoát", command=self.close).pack(side="right")

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(body, text="Luồng xử lý thị giác", padding=8)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.preview_label = ttk.Label(preview_frame, anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        blank = np.full((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), 24, dtype=np.uint8)
        self._show_frame(blank, already_fitted=True)

        result_frame = ttk.LabelFrame(body, text="Thông số phân tích", padding=12)
        result_frame.grid(row=0, column=1, sticky="ns")
        result_frame.configure(width=310)

        ttk.Label(result_frame, textvariable=self.mode_var, wraplength=280).pack(anchor="w")
        ttk.Separator(result_frame).pack(fill="x", pady=10)

        self.status_label = ttk.Label(
            result_frame,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=280,
        )
        self.status_label.pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.confidence_var).pack(anchor="w", pady=(4, 12))

        self.probabilities_frame = ttk.Frame(result_frame)
        self.probabilities_frame.pack(fill="x")

        ttk.Separator(result_frame).pack(fill="x", pady=12)
        ttk.Label(result_frame, text="Mô hình phân loại:").pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.model_var, wraplength=280).pack(anchor="w", pady=(2, 8))

        ttk.Label(result_frame, text="Bộ phát hiện khuôn mặt:").pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.detector_var, wraplength=280).pack(anchor="w", pady=(2, 0))

    def _initialize_models(self) -> None:
        yunet_path = next((p for p in MODEL_YUNET_CANDIDATES if p.is_file()), None)
        if yunet_path is None:
            self.detector_var.set("Thiếu file face_detection_yunet_2023mar.onnx")
            messagebox.showwarning(
                "Thiếu mô hình YuNet",
                "Không tìm thấy file 'face_detection_yunet_2023mar.onnx'. Vui lòng đặt file này cùng thư mục.",
            )
        else:
            try:
                self.face_detector = YuNetFaceDetector(model_path=yunet_path)
                self.detector_var.set(f"YuNet ONNX ({yunet_path.name})")
            except Exception as exc:
                self.detector_var.set("Lỗi nạp YuNet")
                messagebox.showerror("Lỗi Face Detector", str(exc))

        default_classifier = next((p for p in MODEL_CLASSIFIER_CANDIDATES if p.is_file()), None)
        if default_classifier is None:
            self.model_var.set("Chưa nạp. Bấm 'Chọn Classifier Model'.")
            return
        self.load_classifier_model(default_classifier)

    def choose_classifier_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Chọn trọng số mô hình đã huấn luyện",
            initialdir=str(APP_DIR),
            filetypes=[("Keras/HDF5 Model", "*.keras *.h5"), ("Tất cả tệp", "*.*")],
        )
        if selected:
            self.stop_camera()
            self.load_classifier_model(Path(selected))

    def load_classifier_model(self, model_path: Path) -> bool:
        metadata_path = model_path.parent / METADATA_FILENAME
        if not metadata_path.is_file():
            messagebox.showerror(
                "Thiếu Metadata",
                f"Bắt buộc phải có file '{METADATA_FILENAME}' tại:\n{model_path.parent}",
            )
            return False

        self.model_var.set(f"Đang đọc {model_path.name}...")
        self.root.update_idletasks()

        try:
            with metadata_path.open(encoding="utf-8") as f:
                metadata = json.load(f)

            class_names = list(metadata.get("class_names", []))
            if not class_names:
                raise ValueError("Metadata thiếu thuộc tính 'class_names'.")

            model = tf.keras.models.load_model(model_path, compile=False)
            input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
            default_size = tuple(int(v) for v in input_shape[1:3])
            input_size = tuple(int(v) for v in metadata.get("input_size", default_size))
            output_classes = int(model.output_shape[-1])

            if len(class_names) != output_classes:
                raise ValueError(
                    f"Model có {output_classes} outputs nhưng metadata định nghĩa {len(class_names)} lớp."
                )

            # Warm-up inference để biên dịch đồ thị thực thi của TensorFlow, loại bỏ độ trễ ở frame đầu tiên
            dummy = np.zeros((1, *input_size, 3), dtype=np.float32)
            _ = model(dummy, training=False)
        except Exception as exc:
            self.classifier_model = None
            self.model_var.set("Nạp model thất bại")
            messagebox.showerror("Lỗi cấu trúc mô hình", str(exc))
            return False

        self.classifier_model = model
        self.class_names = class_names
        self.input_size = input_size
        self.prediction_history.clear()
        self.last_probabilities = None
        self.model_var.set(f"{model_path.name} ({input_size[0]}x{input_size[1]})")
        self._rebuild_probability_rows()
        self.status_var.set("MÔ HÌNH ĐÃ SẴN SÀNG")
        self.status_label.configure(foreground="#1d4ed8")
        return True

    def _rebuild_probability_rows(self) -> None:
        for child in self.probabilities_frame.winfo_children():
            child.destroy()
        self.class_rows.clear()

        for class_name in self.class_names:
            row = ttk.Frame(self.probabilities_frame)
            row.pack(fill="x", pady=4)
            label_var = tk.StringVar(value=f"{DISPLAY_NAMES.get(class_name, class_name)}: 0.0%")
            ttk.Label(row, textvariable=label_var, style="Class.TLabel").pack(anchor="w")
            progress_var = tk.DoubleVar(value=0.0)
            ttk.Progressbar(row, maximum=100, variable=progress_var).pack(fill="x", pady=(2, 0))
            self.class_rows.append((label_var, progress_var))

    def choose_image(self) -> None:
        if self.classifier_model is None or self.face_detector is None:
            messagebox.showwarning("Cảnh báo", "Cần nạp đủ Classifier Model và YuNet Face Detector trước.")
            return

        selected = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not selected:
            return

        self.stop_camera()
        image_path = Path(selected)
        frame = read_image_unicode(image_path)
        if frame is None:
            messagebox.showerror("Lỗi dữ liệu", f"Không thể đọc file ảnh: {image_path}")
            return

        self.prediction_history.clear()
        faces = self.face_detector.detect(frame)
        display_frame = frame.copy()

        if not faces:
            self.status_var.set("KHÔNG TÌM THẤY MẶT")
            self.status_label.configure(foreground=STATUS_COLORS["neutral"])
            self.confidence_var.set("Độ tin cậy: --")
            self.mode_var.set(f"Ảnh: {image_path.name} (0 phát hiện)")
            self._show_frame(display_frame)
            return

        # Khi xử lý ảnh, chọn khuôn mặt có diện tích lớn nhất (chủ thể chính)
        primary_face = max(faces, key=lambda f: f.area)
        x1, y1, x2, y2 = primary_face.get_padded_box(frame.shape[1], frame.shape[0])
        face_roi = frame[y1:y2, x1:x2]

        try:
            probabilities, inference_ms = self._predict_roi(face_roi)
            self.update_result(probabilities)
            self.mode_var.set(f"Ảnh: {image_path.name} | Inference: {inference_ms:.1f} ms")
            self._render_overlay(display_frame, primary_face, probabilities)
            self._show_frame(display_frame)
        except Exception as exc:
            messagebox.showerror("Lỗi xử lý", str(exc))

    def start_camera(self) -> None:
        if self.classifier_model is None or self.face_detector is None:
            messagebox.showwarning("Cảnh báo", "Yêu cầu nạp đủ Classifier Model và YuNet Face Detector.")
            return
        if self.camera_running:
            return

        camera_idx = int(self.camera_index_var.get())
        if os.name == "nt":
            camera = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(camera_idx)
        else:
            camera = cv2.VideoCapture(camera_idx)

        if not camera.isOpened():
            camera.release()
            messagebox.showerror("Lỗi Camera", f"Không thể kết nối với camera index {camera_idx}.")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.camera = camera
        self.camera_running = True
        self.prediction_history.clear()
        self.last_probabilities = None
        self.last_primary_face = None
        self.last_prediction_time = 0.0
        self.mode_var.set(f"Camera #{camera_idx} đang hoạt động")
        self._read_camera_frame()

    def _read_camera_frame(self) -> None:
        if not self.camera_running or self.camera is None:
            return

        ok, frame = self.camera.read()
        if not ok:
            self.stop_camera()
            messagebox.showerror("Mất tín hiệu", "Không thể đọc khung hình từ cảm biến camera.")
            return

        # Lật ngang khung hình tạo góc nhìn gương giúp tài xế định vị trực quan
        frame = cv2.flip(frame, 1)
        now = time.perf_counter()

        # Thực hiện Face Detection trên từng khung hình để bám sát chuyển động đầu
        faces = self.face_detector.detect(frame)
        if faces:
            # Chọn khuôn mặt chiếm diện tích lớn nhất (người ngồi gần camera nhất - tài xế)
            primary_face = max(faces, key=lambda f: f.area)
            self.last_primary_face = primary_face

            # Giảm tần suất chạy mạng nơ-ron phân loại để duy trì FPS hiển thị mượt mà trên CPU
            if now - self.last_prediction_time >= PREDICTION_INTERVAL_SECONDS:
                x1, y1, x2, y2 = primary_face.get_padded_box(frame.shape[1], frame.shape[0])
                face_roi = frame[y1:y2, x1:x2]

                if face_roi.size > 0:
                    try:
                        probs, inference_ms = self._predict_roi(face_roi)
                        self.prediction_history.append(probs)
                        # Trung bình trượt (Moving Average) để triệt tiêu hiện tượng nhấp nháy xác suất giữa các frame
                        self.last_probabilities = np.mean(self.prediction_history, axis=0)
                        self.last_prediction_time = now
                        self.mode_var.set(
                            f"Live • Face ROI: {x2-x1}x{y2-y1} • {inference_ms:.0f} ms • Lọc {len(self.prediction_history)} frames"
                        )
                        self.update_result(self.last_probabilities)
                    except Exception as exc:
                        self.stop_camera()
                        messagebox.showerror("Lỗi phân loại", str(exc))
                        return
        else:
            self.last_primary_face = None
            if now - self.last_prediction_time >= 1.0:
                self.prediction_history.clear()
                self.last_probabilities = None
                self.status_var.set("KHÔNG TÌM THẤY TÀI XẾ")
                self.status_label.configure(foreground=STATUS_COLORS["neutral"])
                self.confidence_var.set("Độ tin cậy: --")

        display_frame = frame.copy()
        if self.last_primary_face is not None and self.last_probabilities is not None:
            self._render_overlay(display_frame, self.last_primary_face, self.last_probabilities)

        self._show_frame(display_frame)
        self.camera_after_id = self.root.after(CAMERA_REFRESH_MS, self._read_camera_frame)

    def _predict_roi(self, bgr_roi: np.ndarray) -> tuple[np.ndarray, float]:
        if self.classifier_model is None:
            raise RuntimeError("Classifier Model chưa được khởi tạo.")

        rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

        started_at = time.perf_counter()
        output = np.asarray(self.classifier_model(resized[None, ...], training=False))[0].astype(float)
        inference_ms = (time.perf_counter() - started_at) * 1000

        # Kiểm tra lớp cuối nếu model chưa tích hợp hàm kích hoạt Softmax ở output layer
        if (output < 0).any() or not np.isclose(output.sum(), 1.0, atol=1e-3):
            output = tf.nn.softmax(output).numpy()

        if output.shape != (len(self.class_names),):
            raise ValueError(f"Kích thước output ({output.shape}) không khớp số lượng nhãn.")

        return output, inference_ms

    def update_result(self, probabilities: np.ndarray) -> None:
        best_index = int(np.argmax(probabilities))
        best_class = self.class_names[best_index]
        confidence = float(probabilities[best_index])

        if confidence < CONFIDENCE_THRESHOLD:
            self.status_var.set("ĐỘ TIN CẬY THẤP")
            self.status_label.configure(foreground=STATUS_COLORS["neutral"])
        else:
            self.status_var.set(STATUS_MESSAGES.get(best_class, best_class.upper()))
            self.status_label.configure(foreground=STATUS_COLORS.get(best_class, "#1f2937"))

        self.confidence_var.set(f"Độ tin cậy: {confidence:.1%}")

        for index, (label_var, progress_var) in enumerate(self.class_rows):
            score = float(probabilities[index])
            class_name = self.class_names[index]
            label_var.set(f"{DISPLAY_NAMES.get(class_name, class_name)}: {score:.1%}")
            progress_var.set(score * 100)

    def _render_overlay(
        self,
        frame: np.ndarray,
        face: DetectedFace,
        probabilities: np.ndarray,
    ) -> None:
        best_index = int(np.argmax(probabilities))
        best_class = self.class_names[best_index]
        confidence = float(probabilities[best_index])

        if confidence < CONFIDENCE_THRESHOLD:
            color = (160, 160, 160)
            text = f"UNCERTAIN ({confidence:.1%})"
        else:
            color = (40, 180, 40) if best_class == "alert" else (40, 40, 230)
            text = f"{best_class.upper()} ({confidence:.1%})"

        # Vẽ bounding box khuôn mặt
        cv2.rectangle(frame, (face.x, face.y), (face.x + face.w, face.y + face.h), color, 2)

        # Hiển thị thanh trạng thái nhãn tương phản cao ngay trên đỉnh hộp nhận diện
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y1 = max(0, face.y - text_h - 10)
        label_y2 = face.y
        cv2.rectangle(frame, (face.x, label_y1), (face.x + text_w + 8, label_y2), color, -1)
        cv2.putText(
            frame,
            text,
            (face.x + 4, face.y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _show_frame(self, frame: np.ndarray, already_fitted: bool = False) -> None:
        preview = frame if already_fitted else fit_inside(frame, PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self.preview_photo = bgr_to_tk_image(preview)
        self.preview_label.configure(image=self.preview_photo)

    def stop_camera(self) -> None:
        self.camera_running = False
        if self.camera_after_id is not None:
            try:
                self.root.after_cancel(self.camera_after_id)
            except tk.TclError:
                pass
            self.camera_after_id = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.prediction_history.clear()
        self.last_probabilities = None
        self.last_primary_face = None

    def close(self) -> None:
        self.stop_camera()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    DriverStateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()