from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_hunyuan3d_local import (  # noqa: E402
    expected_paint_record,
    expected_shape_record,
    load_omni_point_cloud,
    may_reuse_paint,
    may_reuse_shape,
    patch_omni_dino_loader,
    record_path,
    write_record,
)


class _CpuPipeline:
    device = torch.device("cpu")
    dtype = torch.float32


class OmniPointCloudTests(unittest.TestCase):
    def test_omni_dino_loader_uses_configured_local_snapshot(self) -> None:
        from unittest.mock import patch

        from transformers import Dinov2Model

        local_snapshot = Path("/models/dinov2-large")
        sentinel = object()
        with patch.object(
            Dinov2Model,
            "from_pretrained",
            return_value=sentinel,
        ) as original_loader:
            patch_omni_dino_loader(local_snapshot)
            loaded = Dinov2Model.from_pretrained("facebook/dinov2-large")

        self.assertIs(loaded, sentinel)
        original_loader.assert_called_once_with(
            str(local_snapshot), local_files_only=True
        )

    def test_swaps_axes_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            point_path = Path(directory) / "points_ascii.ply"
            point_path.write_text(
                "\n".join(
                    (
                        "ply",
                        "format ascii 1.0",
                        "element vertex 4",
                        "property float x",
                        "property float y",
                        "property float z",
                        "end_header",
                        "0 0 0",
                        "2 0 0",
                        "0 1 0",
                        "0 0 4",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            points = load_omni_point_cloud(point_path, _CpuPipeline()).numpy()[0]

        self.assertEqual(points.shape, (4, 3))
        np.testing.assert_allclose(points.min(axis=0) + points.max(axis=0), 0.0)
        self.assertTrue(np.isclose(np.ptp(points, axis=0).max(), 1.96))
        # Renderer Z becomes Omni Y; renderer Y becomes Omni Z.
        self.assertGreater(points[3, 1], points[2, 1])
        self.assertGreater(points[2, 2], points[3, 2])

    def _job(self, directory: Path) -> tuple[SimpleNamespace, dict]:
        input_image = directory / "image.png"
        point_cloud = directory / "points.ply"
        untextured = directory / "building_untextured.glb"
        textured = directory / "building.glb"
        input_image.write_bytes(b"image")
        point_cloud.write_bytes(b"points")
        untextured.write_bytes(b"shape")
        textured.write_bytes(b"paint")
        args = SimpleNamespace(
            skip_existing=True,
            shape_backend="omni",
            seed=42,
            shape_steps=50,
            guidance_scale=4.5,
            octree_resolution=512,
            paint_views=6,
            paint_resolution=512,
            no_remesh=False,
        )
        paths = {
            "building_id": 3,
            "output_dir": directory,
            "output_stem": "building",
            "input_image": input_image,
            "point_cloud": point_cloud,
            "omni_dino_path": directory / "dinov2-large",
            "untextured_glb": untextured,
            "textured_glb": textured,
        }
        return args, paths

    def test_omni_resume_requires_matching_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            args, paths = self._job(directory)
            self.assertFalse(may_reuse_shape(args, paths))
            write_record(
                record_path(paths, "shape"), expected_shape_record(args, paths)
            )
            self.assertTrue(may_reuse_shape(args, paths))
            paths["point_cloud"].write_bytes(b"changed points")
            self.assertFalse(may_reuse_shape(args, paths))

    def test_new_shape_cannot_reuse_unrecorded_paint(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            args, paths = self._job(directory)
            self.assertTrue(may_reuse_paint(args, paths))
            write_record(record_path(paths, "shape"), {"backend": "omni"})
            self.assertFalse(may_reuse_paint(args, paths))
            write_record(
                record_path(paths, "paint"), expected_paint_record(args, paths)
            )
            self.assertTrue(may_reuse_paint(args, paths))


if __name__ == "__main__":
    unittest.main()
