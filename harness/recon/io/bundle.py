"""Reading and writing session bundles on disk.

On-disk layout (see PLAN.md §5):

    <bundle>/
      bundle.json         SessionBundle
      images/kf_0000.png  RGB keyframe
      depth/kf_0000.png   uint16 millimetres, 0 == no return (optional)
      masks/kf_0000.png   uint8 semantic label + 1, 0 == no return (optional)
      ground_truth.json   synthetic only
      mesh_gt.ply         synthetic only
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recon.schemas import CANONICAL_FRAME, Keyframe, SessionBundle

#: Depth PNGs are uint16 millimetres. One constant, asserted at both ends.
DEPTH_SCALE_M = 1e-3


class FrameMismatch(RuntimeError):
    """Raised when a bundle's coordinate frame is not the canonical one.

    This is the guard against the flipped-scene failure mode in PLAN.md §7 —
    ARCore's left-handed frame must be converted at the client boundary, not
    silently accepted here.
    """


@dataclass
class BundleDir:
    """A bundle on disk, with lazy per-keyframe image access."""

    root: Path
    bundle: SessionBundle

    @classmethod
    def load(cls, root: str | Path, *, strict_frame: bool = True) -> BundleDir:
        root = Path(root)
        path = root / "bundle.json" if root.is_dir() else root
        bundle = SessionBundle.model_validate_json(Path(path).read_text())
        if strict_frame and not bundle.frame.matches(CANONICAL_FRAME):
            raise FrameMismatch(
                f"bundle {bundle.session_id} is in frame {bundle.frame!r}, "
                f"expected {CANONICAL_FRAME!r}. Convert at the client boundary."
            )
        return cls(root=Path(path).parent, bundle=bundle)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "bundle.json").write_text(self.bundle.model_dump_json(indent=2))

    # -- pixel data ---------------------------------------------------------

    def rgb(self, kf: Keyframe) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.root / kf.image).convert("RGB"))

    def gray(self, kf: Keyframe) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.root / kf.image).convert("L"), dtype=np.float64)

    def depth(self, kf: Keyframe) -> np.ndarray | None:
        """Metric depth in metres, 0 where there is no return."""
        from PIL import Image

        if not kf.depth:
            return None
        p = self.root / kf.depth
        if not p.exists():
            return None
        image = Image.open(p)
        # Mobile depth maps are deliberately stored at the sensor's native low
        # resolution. They are aligned to the RGB frustum, so nearest-neighbour
        # expansion is the correct way to put them on the keyframe pixel grid.
        if image.size != (kf.intrinsics.width, kf.intrinsics.height):
            image = image.resize(
                (kf.intrinsics.width, kf.intrinsics.height), Image.Resampling.NEAREST
            )
        return np.asarray(image, dtype=np.float32) * DEPTH_SCALE_M

    def semantic(self, kf: Keyframe) -> np.ndarray | None:
        """Ground-truth semantic labels (synthetic bundles only), -1 == no return."""
        from PIL import Image

        p = self.root / "masks" / f"{kf.id}.png"
        if not p.exists():
            return None
        return np.asarray(Image.open(p), dtype=np.int32) - 1

    def has_depth(self) -> bool:
        return any(kf.depth for kf in self.bundle.keyframes)

    # -- companions ---------------------------------------------------------

    def ground_truth(self) -> dict | None:
        p = self.root / "ground_truth.json"
        return json.loads(p.read_text()) if p.exists() else None

    def gt_mesh_path(self) -> Path | None:
        p = self.root / "mesh_gt.ply"
        return p if p.exists() else None


def write_depth_png(path: Path, depth_m: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.clip(np.nan_to_num(depth_m) / DEPTH_SCALE_M, 0, 65535).astype(np.uint16)
    Image.fromarray(mm).save(path)
