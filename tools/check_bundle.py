"""Read-only integrity/syntax check; never train or load notebook code.

--write-manifest explicitly records the initial export or an intentional release.
Do not use it to hide a failed integrity check.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "bundle_manifest.json"
BUNDLED_FILES = (
    ".gitattributes", ".gitignore", "README.md", "LICENSE",
    "THIRD_PARTY_NOTICES.md", "licenses/YuNet-LICENSE.txt",
    "app.py", "requirements.txt", "final_training.ipynb",
    "apps/version_3_web/app.py", "apps/version_3_web/requirements.txt",
    "apps/version_3_web/test_app.py", "versions/version_3/class_names.json",
    "versions/version_3/desktop_test_app.py", "versions/version_3/face_app.py",
    "versions/version_3/driver_drowsiness_cnn.keras",
    "versions/version_3/face_detection_yunet_2023mar.onnx",
    "versions/version_3/requirements-desktop.txt",
    "tools/check_bundle.py", "tests/test_bundle.py",
)


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def inventory(root: Path = ROOT) -> dict:
    return {name: file_record(root / name) for name in BUNDLED_FILES}


def validate_integrity(root: Path, manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    expected = manifest.get("files", {})
    if set(expected) != set(BUNDLED_FILES):
        raise ValueError("Manifest file list does not match the bundle contract")
    actual = inventory(root)
    mismatches = [name for name in BUNDLED_FILES if actual[name] != expected[name]]
    if mismatches:
        raise ValueError("Changed files: " + ", ".join(mismatches))


def validate_sources(root: Path = ROOT) -> tuple[int, int]:
    for name in BUNDLED_FILES:
        if name.endswith(".py"):
            ast.parse((root / name).read_text(encoding="utf-8-sig"), filename=name)
    notebook = json.loads((root / "final_training.ipynb").read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError("Expected a v4 notebook")
    code_cells = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"final_training:cell{index}")
            code_cells += 1
    return len(notebook["cells"]), code_cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    try:
        cells, code_cells = validate_sources()
        if args.write_manifest:
            manifest = {
                "schema_version": 1,
                "source_version": "frozen V3 export; no retraining",
                "files": inventory(),
            }
            MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        else:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        validate_integrity(ROOT, manifest)
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as exc:
        parser.exit(1, f"Bundle check failed: {exc}\n")
    print(f"OK: {len(BUNDLED_FILES)} files; notebook {cells} cells/{code_cells} code cells; SHA-256/syntax match.")
    print("This does not verify training quality or a physical camera.")


if __name__ == "__main__":
    main()
