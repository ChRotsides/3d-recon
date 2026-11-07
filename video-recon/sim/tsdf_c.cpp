// extern "C" shim so Python can drive the phone's VoxelHashTSDF through ctypes.
// tsdf.cpp/tsdf.h are compiled unchanged: the simulator fuses with exactly the code the phone runs.
#include "../android/app/src/main/cpp/tsdf.h"

#include <cstring>
#include <vector>

extern "C" {

void *tsdf_create(float voxel, int trunc_voxels, float max_weight, float min_weight, float max_depth) {
  return new VoxelHashTSDF(voxel, trunc_voxels, max_weight, min_weight, max_depth);
}

void tsdf_destroy(void *h) { delete static_cast<VoxelHashTSDF *>(h); }

void tsdf_clear(void *h) { static_cast<VoxelHashTSDF *>(h)->clear(); }

long long tsdf_blocks(void *h) { return (long long)static_cast<VoxelHashTSDF *>(h)->blocks(); }

// stats_out: integrate_ms, extract_ms, blocks_touched, blocks, points
void tsdf_integrate(void *h, const uint16_t *depth, int w, int hgt, const float *K, const float *R, const float *c,
                    const uint8_t *rgb, int rw, int rh, int stride, double *stats_out) {
  auto st = static_cast<VoxelHashTSDF *>(h)->integrate(depth, w, hgt, K, R, c, rgb, rw, rh, stride);
  stats_out[0] = st.integrate_ms;
  stats_out[1] = st.extract_ms;
  stats_out[2] = st.blocks_touched;
  stats_out[3] = st.blocks;
  stats_out[4] = (double)st.points;
}

// Copies at most max_points surface points (x,y,z,r,g,b) into out; returns the number written.
long long tsdf_surface(void *h, float *out, long long max_points) {
  std::vector<float> buf;
  size_t n = static_cast<VoxelHashTSDF *>(h)->surface(buf, (size_t)max_points);
  if (n) std::memcpy(out, buf.data(), n * 6 * sizeof(float));
  return (long long)n;
}
}
