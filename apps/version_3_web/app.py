"""Local Gradio demo for the frozen Version 3 classifier."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import numpy as np
import tensorflow as tf


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = REPOSITORY_DIR / "versions" / "version_3"
MODEL_PATH = MODEL_DIR / "driver_drowsiness_cnn.keras"
METADATA_PATH = MODEL_DIR / "class_names.json"
EXPECTED_CLASSES = ("alert", "drowsy", "yawning")
CONFIDENCE_THRESHOLD = 0.55
SMOOTHING_FRAMES = 5

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


def prepare_rgb_image(image: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    """Match V3 desktop inference after its BGR-to-RGB conversion."""
    if image is None:
        raise ValueError("Chưa có ảnh để kiểm tra.")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) == 0:
        raise ValueError("Ảnh phải có ba kênh màu RGB và kích thước hợp lệ.")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Ảnh chứa giá trị màu không hợp lệ.")
    if array.min() < 0 or array.max() > 255:
        raise ValueError("Giá trị màu ảnh phải nằm trong khoảng 0–255.")
    return cv2.resize(
        array.astype(np.uint8),
        (input_size[1], input_size[0]),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


class V3Predictor:
    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Cần cả model và metadata V3: {model_path} ; {metadata_path}"
            )
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        self.class_names = tuple(metadata.get("class_names", ()))
        if self.class_names != EXPECTED_CLASSES:
            raise ValueError(f"Thứ tự lớp V3 không hợp lệ: {self.class_names}")
        if metadata.get("class_to_index") != {
            name: index for index, name in enumerate(self.class_names)
        }:
            raise ValueError("class_to_index không khớp class_names.")
        if metadata.get("input_color") != "RGB" or metadata.get("input_range") != [0, 255]:
            raise ValueError("Hợp đồng màu/range của model V3 không khớp.")
        self.input_size = tuple(metadata.get("input_size", ()))
        if self.input_size != (224, 224):
            raise ValueError(f"Kích thước input V3 không hợp lệ: {self.input_size}")

        self.model = tf.keras.models.load_model(model_path, compile=False)
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if tuple(input_shape[1:]) != (224, 224, 3):
            raise ValueError(f"Model V3 có input không hợp lệ: {input_shape}")
        if self.model.output_shape[-1] != len(self.class_names):
            raise ValueError("Số output của model không khớp metadata V3.")
        self.model_path = model_path

        # Fail at startup rather than after the first image is submitted.
        self._run_model(np.zeros((224, 224, 3), dtype=np.float32))

    def _run_model(self, prepared_image: np.ndarray) -> np.ndarray:
        output = np.asarray(self.model(prepared_image[None, ...], training=False))[0]
        output = output.astype(np.float64)
        if output.shape != (len(self.class_names),) or not np.isfinite(output).all():
            raise ValueError(f"Output model V3 không hợp lệ: {output.shape}")
        if (output < 0).any() or not np.isclose(output.sum(), 1.0, atol=1e-3):
            output = tf.nn.softmax(output).numpy()
        return output

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        prepared = prepare_rgb_image(image, self.input_size)
        started = time.perf_counter()
        output = self._run_model(prepared)
        return output, (time.perf_counter() - started) * 1000


def format_prediction(probabilities: np.ndarray, inference_ms: float) -> tuple[dict[str, float], str]:
    best_index = int(np.argmax(probabilities))
    best_class = EXPECTED_CLASSES[best_index]
    confidence = float(probabilities[best_index])
    scores = {
        DISPLAY_NAMES[name]: float(probabilities[index])
        for index, name in enumerate(EXPECTED_CLASSES)
    }
    if confidence < CONFIDENCE_THRESHOLD:
        status = "CHƯA ĐỦ CHẮC CHẮN"
    else:
        status = STATUS_MESSAGES[best_class]
    return scores, f"### {status}\nĐộ tin cậy: **{confidence:.1%}** · Xử lý: **{inference_ms:.0f} ms**"


def smooth_prediction(
    probabilities: np.ndarray, history: list[list[float]] | None
) -> tuple[np.ndarray, list[list[float]]]:
    """Return a new history so separate Gradio sessions never share a mutable buffer."""
    updated = (history or [])[-(SMOOTHING_FRAMES - 1) :] + [probabilities.tolist()]
    return np.mean(updated, axis=0), updated


def build_demo(predictor: V3Predictor):
    import gradio as gr

    def predict_image(image):
        if image is None:
            return {}, "Hãy tải một ảnh lên."
        try:
            probabilities, inference_ms = predictor.predict(image)
        except ValueError as exc:
            return {}, f"Không đọc được ảnh: {exc}"
        return format_prediction(probabilities, inference_ms)

    def predict_webcam(image, history):
        if image is None:
            return {}, "Hãy bật webcam.", []
        try:
            probabilities, inference_ms = predictor.predict(image)
        except ValueError as exc:
            return {}, f"Không đọc được khung hình: {exc}", []
        smoothed, updated = smooth_prediction(probabilities, history)
        scores, status = format_prediction(smoothed, inference_ms)
        return scores, status, updated

    def reset_webcam():
        return {}, "Hãy bật webcam.", []

    def reset_image():
        return {}, "Hãy tải một ảnh lên."

    with gr.Blocks(title="Thử model V3 — trạng thái tài xế") as demo:
        gr.Markdown(
            "# Thử model nhận biết trạng thái tài xế · V3\n"
            "Tải ảnh hoặc bật webcam để xem dự đoán **tỉnh táo**, **buồn ngủ / mắt nhắm**, "
            "**đang ngáp**. Model V3 phân loại từng khung hình; kết quả chỉ phục vụ nghiên cứu "
            "và không thay thế thiết bị cảnh báo an toàn."
        )
        with gr.Tab("Tải ảnh"):
            with gr.Row():
                uploaded_image = gr.Image(
                    label="Ảnh cần kiểm tra", sources=["upload"], type="numpy", image_mode="RGB"
                )
                with gr.Column():
                    image_status = gr.Markdown("Hãy tải một ảnh lên.")
                    image_scores = gr.Label(label="Xác suất ba trạng thái", num_top_classes=3)
            gr.Button("Kiểm tra ảnh", variant="primary").click(
                predict_image,
                inputs=uploaded_image,
                outputs=[image_scores, image_status],
                concurrency_limit=1,
                concurrency_id="v3_inference",
            )
            uploaded_image.clear(reset_image, outputs=[image_scores, image_status])

        with gr.Tab("Webcam"):
            gr.Markdown("Cho phép trình duyệt sử dụng camera, sau đó bấm nút ghi hình trong khung webcam.")
            history = gr.State(value=[])
            with gr.Row():
                webcam = gr.Image(
                    label="Webcam", sources=["webcam"], type="numpy", image_mode="RGB", streaming=True
                )
                with gr.Column():
                    webcam_status = gr.Markdown("Hãy bật webcam.")
                    webcam_scores = gr.Label(label="Xác suất ba trạng thái", num_top_classes=3)
            webcam.stream(
                predict_webcam,
                inputs=[webcam, history],
                outputs=[webcam_scores, webcam_status, history],
                stream_every=0.5,
                time_limit=3600,
                concurrency_limit=1,
                concurrency_id="v3_inference",
            )
            webcam.clear(
                reset_webcam,
                outputs=[webcam_scores, webcam_status, history],
            )
            gr.Button("Xóa kết quả webcam").click(
                reset_webcam,
                outputs=[webcam_scores, webcam_status, history],
            )

    demo.queue(max_size=8)
    return demo


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Website local thử ảnh và webcam với model V3.")
    parser.add_argument("--port", type=int, default=7860, help="Cổng localhost (mặc định 7860).")
    parser.add_argument("--no-browser", action="store_true", help="Không tự mở trình duyệt.")
    parser.add_argument("--check", action="store_true", help="Kiểm tra model và giao diện, không mở máy chủ.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Cổng phải nằm trong khoảng 1–65535.")
    predictor = V3Predictor()
    demo = build_demo(predictor)
    if args.check:
        print("Model V3 và giao diện đã sẵn sàng: RGB 224×224, alert/drowsy/yawning.")
        return
    demo.launch(
        server_name="127.0.0.1", server_port=args.port,
        share=False, inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
