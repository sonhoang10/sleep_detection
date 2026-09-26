"""Portable bundle checks; only the CLI test imports the inference dependencies."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest

from tools.check_bundle import MANIFEST, ROOT, validate_integrity, validate_sources


class TestBundle(unittest.TestCase):
    def test_project_uses_single_layout(self):
        self.assertFalse((ROOT / "versions").exists())
        self.assertTrue((ROOT / "models").is_dir())
        self.assertEqual((ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0], "# Sleep Detection")
        self.assertIn(
            'title="Sleep Detection — trạng thái tài xế"',
            (ROOT / "apps/app.py").read_text(encoding="utf-8"),
        )

    def test_shipped_files_match_manifest(self):
        validate_integrity(ROOT, json.loads(MANIFEST.read_text(encoding="utf-8")))

    def test_changed_checksum_is_rejected(self):
        manifest = copy.deepcopy(json.loads(MANIFEST.read_text(encoding="utf-8")))
        manifest["files"]["models/driver_drowsiness_cnn.keras"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Changed files"):
            validate_integrity(ROOT, manifest)

    def test_original_notebook_and_source_syntax(self):
        self.assertEqual(validate_sources(), (32, 17))

    def test_optional_yunet_loads_and_resizes(self):
        import numpy as np
        from apps.face_app import YuNetFaceDetector

        detector = YuNetFaceDetector(ROOT / "models/face_detection_yunet_2023mar.onnx")
        for width, height in ((320, 320), (640, 480)):
            with self.subTest(size=(width, height)):
                self.assertEqual(detector.detect(np.zeros((height, width, 3), dtype=np.uint8)), [])

    def test_entrypoint_works_outside_repository(self):
        with tempfile.TemporaryDirectory(prefix="sleep-detection-portability-") as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "app.py"), "--check"],
                cwd=directory, capture_output=True, encoding="utf-8", timeout=120,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("alert/drowsy/yawning", result.stdout)


if __name__ == "__main__":
    unittest.main()
