r"""Ứng dụng desktop thử model bằng ảnh hoặc camera máy tính.

Chạy:
    .\.venv\Scripts\python.exe .\apps\desktop_test_app.py

Đặt ``driver_drowsiness_cnn.keras`` và ``class_names.json`` trong thư mục ``models/``,
hoặc dùng nút "Chọn model" khi ứng dụng mở.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import tkinter as tk
from collections import deque
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
        "Hãy mở PowerShell trong thư mục dự án và chạy:\n"
        "py -3.13 -m venv .venv\n"
        r".\.venv\Scripts\python.exe -m pip install -r .\requirements.txt"
        "\nSau đó chạy:\n"
        r".\.venv\Scripts\python.exe .\apps\desktop_test_app.py"
    ) from exc


APP_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_CANDIDATES = (
    APP_DIR / "driver_drowsiness_cnn.keras",
    APP_DIR / "best_finetuned_model.keras",
    APP_DIR / "driver_drowsiness_cnn.h5",
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
}

CONFIDENCE_THRESHOLD = 0.55
SMOOTHING_FRAMES = 5
PREDICTION_INTERVAL_SECONDS = 0.25
CAMERA_REFRESH_MS = 30
PREVIEW_WIDTH = 820
PREVIEW_HEIGHT = 560


def read_image_unicode(path: Path):
    """Đọc được cả đường dẫn Windows có ký tự Unicode."""
    encoded_path = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(encoded_path, cv2.IMREAD_COLOR)


def fit_inside(frame, width: int, height: int):
    """Thu ảnh đúng tỉ lệ và đặt giữa nền tối."""
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


def bgr_to_tk_image(frame):
    """Chuyển frame OpenCV thành ảnh mà Tkinter hiển thị được."""
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Không thể chuyển frame để hiển thị.")
    png_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return tk.PhotoImage(data=png_base64)


class DriverStateApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Thử model nhận biết trạng thái tài xế")
        self.root.minsize(1120, 700)

        self.model = None
        self.model_path: Path | None = None
        self.class_names: list[str] = []
        self.input_size = (224, 224)

        self.camera = None
        self.camera_running = False
        self.camera_after_id = None
        self.last_prediction_time = 0.0
        self.last_probabilities = None
        self.prediction_history = deque(maxlen=SMOOTHING_FRAMES)

        self.preview_photo = None
        self.class_rows = []

        self.mode_var = tk.StringVar(value="Chưa chọn ảnh hoặc bật camera")
        self.model_var = tk.StringVar(value="Chưa nạp model")
        self.status_var = tk.StringVar(value="CHƯA CÓ KẾT QUẢ")
        self.confidence_var = tk.StringVar(value="Độ tin cậy: --")
        self.camera_index_var = tk.IntVar(value=0)

        self._build_interface()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._load_default_model)

    def _build_interface(self):
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Class.TLabel", font=("Segoe UI", 10))

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Nhận biết trạng thái tài xế",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Thử bằng ảnh hoặc camera máy tính. Giữ khuôn mặt rõ, đủ sáng và gần camera. "
                "Đây là bản thử nghiệm nghiên cứu, không phải thiết bị cảnh báo an toàn."
            ),
            wraplength=1050,
        ).pack(anchor="w", pady=(2, 12))

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 10))

        ttk.Button(controls, text="Chọn model", command=self.choose_model).pack(side="left")
        ttk.Button(controls, text="Chọn ảnh", command=self.choose_image).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(controls, text="Bật camera", command=self.start_camera).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(controls, text="Dừng camera", command=self.stop_camera).pack(
            side="left", padx=(8, 0)
        )

        ttk.Label(controls, text="Camera số:").pack(side="left", padx=(20, 5))
        ttk.Spinbox(
            controls,
            from_=0,
            to=5,
            width=4,
            textvariable=self.camera_index_var,
        ).pack(side="left")

        ttk.Button(controls, text="Thoát", command=self.close).pack(side="right")

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(body, text="Hình ảnh", padding=8)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.preview_label = ttk.Label(preview_frame, anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        blank = np.full((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), 24, dtype=np.uint8)
        cv2.putText(
            blank,
            "Select an image or start the camera",
            (175, PREVIEW_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        self._show_frame(blank, already_fitted=True)

        result_frame = ttk.LabelFrame(body, text="Kết quả", padding=12)
        result_frame.grid(row=0, column=1, sticky="ns")
        result_frame.configure(width=290)

        ttk.Label(result_frame, textvariable=self.mode_var, wraplength=270).pack(anchor="w")
        ttk.Separator(result_frame).pack(fill="x", pady=10)

        self.status_label = ttk.Label(
            result_frame,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=270,
        )
        self.status_label.pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.confidence_var).pack(
            anchor="w", pady=(4, 14)
        )

        self.probabilities_frame = ttk.Frame(result_frame)
        self.probabilities_frame.pack(fill="x")

        ttk.Separator(result_frame).pack(fill="x", pady=12)
        ttk.Label(result_frame, text="Model đang dùng:").pack(anchor="w")
        ttk.Label(
            result_frame,
            textvariable=self.model_var,
            wraplength=270,
        ).pack(anchor="w", pady=(3, 0))

        ttk.Label(
            result_frame,
            text=(
                "Nếu kết quả thay đổi liên tục, hãy tăng ánh sáng, đưa mặt gần hơn "
                "hoặc thử bằng ảnh tĩnh trước."
            ),
            wraplength=270,
        ).pack(anchor="w", pady=(18, 0))

    def _load_default_model(self):
        default_model = next((path for path in MODEL_CANDIDATES if path.is_file()), None)
        if default_model is None:
            self.model_var.set("Không thấy model cạnh file ứng dụng. Bấm 'Chọn model'.")
            return
        self.load_model(default_model)

    def choose_model(self):
        selected = filedialog.askopenfilename(
            title="Chọn model đã train",
            initialdir=str(APP_DIR),
            filetypes=[
                ("Keras model", "*.keras"),
                ("HDF5 model", "*.h5"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if selected:
            self.stop_camera()
            self.load_model(Path(selected))

    def load_model(self, model_path: Path):
        metadata_path = model_path.parent / METADATA_FILENAME
        if not metadata_path.is_file():
            messagebox.showerror(
                "Thiếu class_names.json",
                f"Hãy đặt {METADATA_FILENAME} cùng thư mục với model:\n{model_path.parent}",
            )
            return False

        self.model_var.set(f"Đang nạp {model_path.name}...")
        self.root.update_idletasks()

        try:
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)

            class_names = list(metadata.get("class_names", []))
            if not class_names:
                raise ValueError("class_names.json không có danh sách 'class_names'.")

            model = tf.keras.models.load_model(model_path, compile=False)
            input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
            default_size = tuple(int(value) for value in input_shape[1:3])
            input_size = tuple(int(value) for value in metadata.get("input_size", default_size))
            output_classes = int(model.output_shape[-1])

            if len(input_size) != 2 or any(value <= 0 for value in input_size):
                raise ValueError(f"Kích thước input không hợp lệ: {input_size}")
            if len(class_names) != output_classes:
                raise ValueError(
                    f"Model có {output_classes} output nhưng metadata có "
                    f"{len(class_names)} lớp: {class_names}"
                )

            class_to_index = metadata.get("class_to_index", {})
            if class_to_index:
                expected_mapping = {name: index for index, name in enumerate(class_names)}
                normalized_mapping = {name: int(index) for name, index in class_to_index.items()}
                if normalized_mapping != expected_mapping:
                    raise ValueError(
                        "Thứ tự class_to_index không khớp class_names trong metadata."
                    )

            dummy = np.zeros((1, *input_size, 3), dtype=np.float32)
            dummy_output = np.asarray(model(dummy, training=False))
            if dummy_output.shape != (1, output_classes):
                raise ValueError(f"Output model không hợp lệ: {dummy_output.shape}")
        except Exception as exc:
            self.model = None
            self.model_var.set("Nạp model thất bại")
            messagebox.showerror("Không thể nạp model", str(exc))
            return False

        self.model = model
        self.model_path = model_path
        self.class_names = class_names
        self.input_size = input_size
        self.prediction_history.clear()
        self.last_probabilities = None
        self.model_var.set(
            f"{model_path.name}\nInput: {input_size[0]}×{input_size[1]} RGB\n"
            f"Lớp: {', '.join(class_names)}"
        )
        self._rebuild_probability_rows()
        self.status_var.set("MODEL ĐÃ SẴN SÀNG")
        self.confidence_var.set("Độ tin cậy: --")
        self.status_label.configure(foreground="#1d4ed8")
        return True

    def _rebuild_probability_rows(self):
        for child in self.probabilities_frame.winfo_children():
            child.destroy()
        self.class_rows.clear()

        for class_name in self.class_names:
            row = ttk.Frame(self.probabilities_frame)
            row.pack(fill="x", pady=4)
            label_var = tk.StringVar(value=f"{DISPLAY_NAMES.get(class_name, class_name)}: 0,0%")
            ttk.Label(row, textvariable=label_var, style="Class.TLabel").pack(anchor="w")
            progress_var = tk.DoubleVar(value=0.0)
            ttk.Progressbar(row, maximum=100, variable=progress_var).pack(fill="x", pady=(2, 0))
            self.class_rows.append((label_var, progress_var))

    def _require_model(self):
        if self.model is not None:
            return True
        messagebox.showinfo(
            "Chưa có model",
            "Hãy bấm 'Chọn model' và chọn driver_drowsiness_cnn.keras trước.",
        )
        self.choose_model()
        return self.model is not None

    def choose_image(self):
        if not self._require_model():
            return

        selected = filedialog.askopenfilename(
            title="Chọn ảnh cần kiểm tra",
            filetypes=[
                ("Tệp ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if not selected:
            return

        self.stop_camera()
        image_path = Path(selected)
        frame = read_image_unicode(image_path)
        if frame is None:
            messagebox.showerror("Không đọc được ảnh", str(image_path))
            return

        self.prediction_history.clear()
        try:
            probabilities, inference_ms = self.predict(frame)
        except Exception as exc:
            messagebox.showerror("Dự đoán thất bại", str(exc))
            return

        self.mode_var.set(f"Ảnh: {image_path.name} • xử lý {inference_ms:.0f} ms")
        self.update_result(probabilities)
        self._show_frame(self.draw_prediction(frame.copy(), probabilities))

    def start_camera(self):
        if not self._require_model() or self.camera_running:
            return

        camera_index = int(self.camera_index_var.get())
        if os.name == "nt":
            camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(camera_index)
        else:
            camera = cv2.VideoCapture(camera_index)

        if not camera.isOpened():
            camera.release()
            messagebox.showerror(
                "Không mở được camera",
                f"Không mở được camera số {camera_index}. Hãy đóng ứng dụng khác đang dùng camera "
                "hoặc thử camera số khác.",
            )
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera = camera
        self.camera_running = True
        self.prediction_history.clear()
        self.last_probabilities = None
        self.last_prediction_time = 0.0
        self.mode_var.set(f"Camera trực tiếp • camera số {camera_index}")
        self._read_camera_frame()

    def _read_camera_frame(self):
        if not self.camera_running or self.camera is None:
            return

        ok, frame = self.camera.read()
        if not ok:
            self.stop_camera()
            messagebox.showerror("Mất camera", "Không đọc được khung hình từ camera.")
            return

        now = time.perf_counter()
        if now - self.last_prediction_time >= PREDICTION_INTERVAL_SECONDS:
            try:
                current_probabilities, inference_ms = self.predict(frame)
                self.prediction_history.append(current_probabilities)
                self.last_probabilities = np.mean(self.prediction_history, axis=0)
                self.last_prediction_time = now
                self.mode_var.set(
                    f"Camera trực tiếp • xử lý {inference_ms:.0f} ms • "
                    f"làm mượt {len(self.prediction_history)}/{SMOOTHING_FRAMES} frame"
                )
                self.update_result(self.last_probabilities)
            except Exception as exc:
                self.stop_camera()
                messagebox.showerror("Dự đoán camera thất bại", str(exc))
                return

        display_frame = frame.copy()
        if self.last_probabilities is not None:
            display_frame = self.draw_prediction(display_frame, self.last_probabilities)
        self._show_frame(display_frame)
        self.camera_after_id = self.root.after(CAMERA_REFRESH_MS, self._read_camera_frame)

    def stop_camera(self):
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

    def predict(self, frame):
        if self.model is None:
            raise RuntimeError("Model chưa được nạp.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

        started_at = time.perf_counter()
        output = np.asarray(self.model(resized[None, ...], training=False))[0].astype(float)
        inference_ms = (time.perf_counter() - started_at) * 1000

        if (output < 0).any() or not np.isclose(output.sum(), 1.0, atol=1e-3):
            output = tf.nn.softmax(output).numpy()
        if output.shape != (len(self.class_names),):
            raise ValueError(f"Output model không đúng kích thước: {output.shape}")
        return output, inference_ms

    def update_result(self, probabilities):
        best_index = int(np.argmax(probabilities))
        best_class = self.class_names[best_index]
        confidence = float(probabilities[best_index])

        if confidence < CONFIDENCE_THRESHOLD:
            self.status_var.set("MODEL CHƯA ĐỦ CHẮC CHẮN")
            self.status_label.configure(foreground="#6b7280")
        else:
            self.status_var.set(STATUS_MESSAGES.get(best_class, best_class.upper()))
            self.status_label.configure(foreground=STATUS_COLORS.get(best_class, "#1f2937"))
        self.confidence_var.set(f"Độ tin cậy: {confidence:.1%}")

        for index, (label_var, progress_var) in enumerate(self.class_rows):
            score = float(probabilities[index])
            class_name = self.class_names[index]
            label_var.set(f"{DISPLAY_NAMES.get(class_name, class_name)}: {score:.1%}")
            progress_var.set(score * 100)

    def draw_prediction(self, frame, probabilities):
        best_index = int(np.argmax(probabilities))
        best_class = self.class_names[best_index]
        confidence = float(probabilities[best_index])

        if confidence < CONFIDENCE_THRESHOLD:
            text = f"UNCERTAIN  {confidence:.1%}"
            color = (160, 160, 160)
        else:
            text = f"{best_class.upper()}  {confidence:.1%}"
            color = (40, 180, 40) if best_class == "alert" else (40, 40, 230)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 66), (20, 20, 20), -1)
        cv2.putText(
            frame,
            text,
            (20, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
        return frame

    def _show_frame(self, frame, already_fitted=False):
        preview = frame if already_fitted else fit_inside(frame, PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self.preview_photo = bgr_to_tk_image(preview)
        self.preview_label.configure(image=self.preview_photo)

    def close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    DriverStateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
