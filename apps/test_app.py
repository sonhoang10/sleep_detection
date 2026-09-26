"""Focused checks for the local web wrapper."""

from __future__ import annotations

import unittest
import io
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from apps.app import (
    CONFIDENCE_THRESHOLD,
    SMOOTHING_FRAMES,
    Predictor,
    build_demo,
    format_prediction,
    prepare_rgb_image,
    smooth_prediction,
    main,
)
from apps.desktop_test_app import DriverStateApp


class TestWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor()

    def test_default_launch_automatically_selects_local_port(self):
        demo = Mock()
        with patch("sys.argv", ["app.py", "--no-browser"]), patch(
            "apps.app.Predictor"
        ), patch("apps.app.build_demo", return_value=demo):
            main()
        demo.launch.assert_called_once_with(
            server_name="127.0.0.1", server_port=None, share=False, inbrowser=False,
        )

    def test_explicit_busy_port_has_actionable_error(self):
        demo = Mock()
        demo.launch.side_effect = OSError("Cannot find empty port in range: 7860-7860")
        stderr = io.StringIO()
        with patch("sys.argv", ["app.py", "--port", "7860"]), patch(
            "apps.app.Predictor"
        ), patch("apps.app.build_demo", return_value=demo), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as result:
                main()
        self.assertEqual(result.exception.code, 1)
        self.assertIn("Cổng 7860 đang bận", stderr.getvalue())
        self.assertIn("Bỏ --port", stderr.getvalue())
        self.assertEqual(demo.launch.call_args.kwargs["server_port"], 7860)

    def test_invalid_port_is_rejected_before_loading_model(self):
        for port in ("0", "65536"):
            with self.subTest(port=port), patch("sys.argv", ["app.py", "--port", port]), patch(
                "apps.app.Predictor"
            ) as load_model, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main()
                self.assertEqual(result.exception.code, 2)
                load_model.assert_not_called()

    def test_real_model_matches_frozen_desktop_inference(self):
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        web_probabilities, _ = self.predictor.predict(rgb)
        desktop_context = SimpleNamespace(
            model=self.predictor.model,
            input_size=self.predictor.input_size,
            class_names=list(self.predictor.class_names),
        )
        desktop_probabilities, _ = DriverStateApp.predict(desktop_context, bgr)

        self.assertEqual(self.predictor.class_names, ("alert", "drowsy", "yawning"))
        self.assertEqual(web_probabilities.shape, (3,))
        self.assertAlmostEqual(float(web_probabilities.sum()), 1.0, places=5)
        np.testing.assert_allclose(web_probabilities, desktop_probabilities, atol=1e-6)

    def test_preprocessing_and_invalid_images(self):
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        self.assertEqual(prepare_rgb_image(rgb, (224, 224)).shape, (224, 224, 3))
        with self.assertRaisesRegex(ValueError, "ba kênh"):
            prepare_rgb_image(np.zeros((4, 6), dtype=np.uint8), (224, 224))
        with self.assertRaisesRegex(ValueError, "0–255"):
            prepare_rgb_image(np.full((4, 6, 3), 300.0), (224, 224))
        with self.assertRaisesRegex(ValueError, "không hợp lệ"):
            prepare_rgb_image(np.full((4, 6, 3), np.nan), (224, 224))

    def test_uncertain_status_and_probability_labels(self):
        scores, status = format_prediction(
            np.array([CONFIDENCE_THRESHOLD - 0.01, 0.30, 0.16]), 12.0
        )
        self.assertIn("CHƯA ĐỦ CHẮC CHẮN", status)
        self.assertEqual(list(scores), ["Tỉnh táo", "Buồn ngủ / mắt nhắm", "Đang ngáp"])
        self.assertEqual(len(scores), 3)

    def test_webcam_history_is_bounded_and_session_local(self):
        session_a = None
        session_b = None
        for _ in range(SMOOTHING_FRAMES + 2):
            mean_a, session_a = smooth_prediction(np.array([1.0, 0.0, 0.0]), session_a)
        mean_b, session_b = smooth_prediction(np.array([0.0, 1.0, 0.0]), session_b)
        self.assertEqual(len(session_a), SMOOTHING_FRAMES)
        self.assertEqual(len(session_b), 1)
        np.testing.assert_array_equal(mean_a, [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(mean_b, [0.0, 1.0, 0.0])
        self.assertIsNot(session_a, session_b)

    def test_missing_model_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            Predictor(Path("missing.keras"), Path("missing.json"))

    def test_gradio_image_and_camera_callbacks(self):
        demo = build_demo(self.predictor)
        callbacks = {value.fn.__name__: value.fn for value in demo.fns.values()}
        image_fn = callbacks["predict_image"]
        camera_fn = callbacks["predict_webcam"]
        self.assertEqual(image_fn(None), ({}, "Hãy tải một ảnh lên."))
        self.assertEqual(callbacks["reset_image"](), ({}, "Hãy tải một ảnh lên."))
        scores, status = image_fn(np.zeros((4, 6), dtype=np.uint8))
        self.assertEqual(scores, {})
        self.assertIn("Không đọc được ảnh", status)
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        scores, _, history = camera_fn(image, [])
        self.assertEqual(len(scores), 3)
        self.assertEqual(len(history), 1)
        self.assertEqual(camera_fn(None, history)[2], [])
        self.assertEqual(callbacks["reset_webcam"]()[2], [])


if __name__ == "__main__":
    unittest.main()
