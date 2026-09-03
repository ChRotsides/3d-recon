"""Video + phone sensors -> room geometry, without ARCore.

The phone records ordinary video, the raw inertial and magnetic sensors, and the
camera's own intrinsic calibration. Offline, the sensors give every frame its
orientation, the gravity direction and the metric scale; structure-from-motion on
the video gives position; Depth Anything 3 gives dense depth conditioned on
those poses; TSDF fusion turns the depth into a mesh and a point cloud.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
