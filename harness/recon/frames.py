"""The one coordinate convention in this repo, and the one conversion out of it.

PLAN.md §5: right-handed, **Y up**, world<-camera poses, the camera looking down
**-Z** with +X right and +Y up. That is the OpenGL convention, and it is what
every pose in a `SessionBundle` means.

Almost everything downstream — DA3, OpenCV, Open3D — is the *other* one: +Y
down, +Z forward. The conversion between them is a diagonal sign flip, which is
exactly why it is dangerous: applying it twice, or not at all, produces poses
that are still valid rotations pointing somewhere plausible. Nothing throws. The
geometry is quietly mirrored, and PLAN.md §7 lists that as a failure mode
precisely because only a measurement catches it.

Measured on a real capture: handing DA3 unconverted extrinsics spread the
per-frame scale ratio 2.34x against the bundle's own sparse points; converted,
1.16x. It reads as a bad depth model rather than as a frame bug.

Live in one place so there is one thing to be right.
"""

from __future__ import annotations

import numpy as np

#: OpenGL (+Y up, -Z forward) -> OpenCV (+Y down, +Z forward). Its own inverse.
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def world_to_camera_cv(kf) -> np.ndarray:
    """A keyframe's pose as the 4x4 world-to-camera matrix in OpenCV convention.

    This is what DA3 wants as `extrinsics` and what Open3D wants as `extrinsic`.
    """
    return np.linalg.inv(kf.pose.matrix() @ GL_TO_CV)
