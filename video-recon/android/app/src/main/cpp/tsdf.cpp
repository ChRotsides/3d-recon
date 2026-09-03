#include "tsdf.h"
#include <chrono>
#include <algorithm>
#include <array>

static double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

VoxelHashTSDF::VoxelHashTSDF(float voxel, int truncVoxels, float maxWeight, float minWeight, float maxDepth)
    : voxel_(voxel), trunc_(truncVoxels * voxel), maxW_(maxWeight), minW_(minWeight), maxDepth_(maxDepth), blockSize_(8 * voxel) {}

TsdfBlock& VoxelHashTSDF::block(int64_t k, int bx, int by, int bz) {
    auto it = blocks_.find(k);
    if (it != blocks_.end()) return it->second;
    TsdfBlock& b = blocks_[k]; b.bx = bx; b.by = by; b.bz = bz;
    std::fill(b.tsdf, b.tsdf + 512, 1.0f); std::fill(b.w, b.w + 512, 0.0f); std::fill(b.rgb, b.rgb + 1536, 0);
    return b;
}

bool VoxelHashTSDF::flying(const uint16_t* d, int w, int h, int u, int v) {
    const int c = d[v * w + u]; const int th = (int)(0.08f * c);
    if (u > 0 && std::abs(d[v * w + u - 1] - c) > th) return true;
    if (u < w - 1 && std::abs(d[v * w + u + 1] - c) > th) return true;
    if (v > 0 && std::abs(d[(v - 1) * w + u] - c) > th) return true;
    if (v < h - 1 && std::abs(d[(v + 1) * w + u] - c) > th) return true;
    return false;
}

VoxelHashTSDF::Stats VoxelHashTSDF::integrate(const uint16_t* depth, int w, int h, const float K[4], const float R[9], const float c[3],
                                              const uint8_t* rgb, int rw, int rh, int stride) {
    Stats st; const double t0 = nowMs();
    const float fx = K[0], fy = K[1], cx = K[2], cy = K[3], T = trunc_, V = voxel_, BS = blockSize_;
    std::unordered_set<int64_t> band, carve; const float step = BS * 0.5f; const int phase = frames_ & 3;
    for (int v = 0; v < h; v += stride) for (int u = 0; u < w; u += stride) {
        const float z = depth[v * w + u] / 1000.0f; if (z <= 0 || z > maxDepth_) continue;
        if (flying(depth, w, h, u, v)) continue;
        const float dx = (u - cx) / fx, dy = (v - cy) / fy; const float n = std::sqrt(dx * dx + dy * dy + 1);
        const float rx = (R[0] * dx + R[1] * dy + R[2]) / n, ry = (R[3] * dx + R[4] * dy + R[5]) / n, rz = (R[6] * dx + R[7] * dy + R[8]) / n;
        const float r0 = std::max(0.05f, (z - T) * n), r1 = (z + T) * n;
        for (float r = r0; r <= r1 + 1e-6f; r += step) {
            const float px = c[0] + rx * r, py = c[1] + ry * r, pz = c[2] + rz * r;
            band.insert(key((int)std::floor(px / BS), (int)std::floor(py / BS), (int)std::floor(pz / BS)));
        }
        if (((u / stride) + (v / stride)) % 2 == 0) for (float r = 0.1f; r < r0; r += BS) {
            const float px = c[0] + rx * r, py = c[1] + ry * r, pz = c[2] + rz * r;
            const int64_t k = key((int)std::floor(px / BS), (int)std::floor(py / BS), (int)std::floor(pz / BS));
            if (blocks_.count(k)) carve.insert(k);
        }
    }
    std::vector<std::pair<int64_t, bool>> work; work.reserve(band.size() + carve.size());
    for (auto k : band) work.push_back({k, false});
    for (auto k : carve) if (!band.count(k)) work.push_back({k, true});
    st.blocks_touched = (int)work.size();
    for (auto& item : work) {
        const int64_t k = item.first; const bool sub = item.second;
        int bx, by, bz; unkey(k, bx, by, bz); TsdfBlock& b = block(k, bx, by, bz);
        const float ox = bx * BS + V / 2 - c[0], oy = by * BS + V / 2 - c[1], oz = bz * BS + V / 2 - c[2]; bool changed = false;
        for (int i = 0, idx = 0; i < 8; i++) for (int j = 0; j < 8; j++) for (int l = 0; l < 8; l++, idx++) {
            if (sub && ((i + j + l) & 3) != phase) continue;
            const float px = ox + i * V, py = oy + j * V, pz = oz + l * V;
            const float zc = R[2] * px + R[5] * py + R[8] * pz; if (zc <= 0.05f) continue;
            const float xc = R[0] * px + R[3] * py + R[6] * pz, yc = R[1] * px + R[4] * py + R[7] * pz;
            const int u = (int)std::lround(fx * xc / zc + cx), v = (int)std::lround(fy * yc / zc + cy); if (u < 0 || v < 0 || u >= w || v >= h) continue;
            const float d = depth[v * w + u] / 1000.0f; if (d <= 0 || d > maxDepth_) continue;
            const float sdf = d - zc; if (sdf < -T) continue;
            const float t = sdf > T ? 1.0f : sdf / T; const float wo = b.w[idx], wn = std::min(maxW_, wo + 1);
            b.tsdf[idx] = (b.tsdf[idx] * wo + t) / (wo + 1); b.w[idx] = wn; changed = true;
            if (rgb && std::fabs(sdf) < T * 0.5f) {
                const int ru = std::min(rw - 1, (int)std::lround(u * (float)rw / w)), rv = std::min(rh - 1, (int)std::lround(v * (float)rh / h)); const int q = 4 * (rv * rw + ru), o = idx * 3;
                for (int ch = 0; ch < 3; ch++) b.rgb[o + ch] = (uint8_t)((b.rgb[o + ch] * wo + rgb[q + ch]) / (wo + 1));
            }
        }
        if (changed) { b.dirty = true; touched_.insert(k); }
    }
    frames_++; st.integrate_ms = nowMs() - t0;
    const double t1 = nowMs(); std::unordered_set<int64_t> redo(touched_);
    const std::array<std::array<int, 3>, 3> nbrs = {{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}};
    for (auto k : touched_) { int bx, by, bz; unkey(k, bx, by, bz);
        for (auto& d : nbrs) { const int64_t kk = key(bx + d[0], by + d[1], bz + d[2]); if (blocks_.count(kk)) redo.insert(kk); } }
    for (auto k : redo) { auto it = blocks_.find(k); if (it != blocks_.end()) extractBlock(it->second); }
    touched_.clear(); st.extract_ms = nowMs() - t1; st.blocks = (int)blocks_.size();
    for (auto& kv : blocks_) st.points += kv.second.pts.size() / 6;
    return st;
}

const TsdfBlock* VoxelHashTSDF::neighbour(int bx, int by, int bz, int& i, int& j, int& l, int& idx) const {
    if (i >= 8) { bx++; i = 0; } if (j >= 8) { by++; j = 0; } if (l >= 8) { bz++; l = 0; }
    auto it = blocks_.find(key(bx, by, bz)); if (it == blocks_.end()) return nullptr; idx = (i * 8 + j) * 8 + l; return &it->second;
}

void VoxelHashTSDF::extractBlock(TsdfBlock& b) {
    const float V = voxel_, BS = blockSize_; b.pts.clear();
    const float ox = b.bx * BS + V / 2, oy = b.by * BS + V / 2, oz = b.bz * BS + V / 2;
    for (int i = 0, idx = 0; i < 8; i++) for (int j = 0; j < 8; j++) for (int l = 0; l < 8; l++, idx++) {
        const float w0 = b.w[idx]; if (w0 < minW_) continue; const float t0 = b.tsdf[idx];
        for (int ax = 0; ax < 3; ax++) {
            int ni = i + (ax == 0), nj = j + (ax == 1), nl = l + (ax == 2), idx1 = 0;
            const TsdfBlock* nb = neighbour(b.bx, b.by, b.bz, ni, nj, nl, idx1); if (!nb) continue;
            if (nb->w[idx1] < minW_) continue; const float t1 = nb->tsdf[idx1];
            if ((t0 < 0) == (t1 < 0) || std::fabs(t0 - t1) < 1e-6f) continue;
            const float a = t0 / (t0 - t1);
            b.pts.push_back(ox + i * V + (ax == 0 ? a * V : 0)); b.pts.push_back(oy + j * V + (ax == 1 ? a * V : 0)); b.pts.push_back(oz + l * V + (ax == 2 ? a * V : 0));
            const int o0 = idx * 3, o1 = idx1 * 3;
            for (int ch = 0; ch < 3; ch++) b.pts.push_back((b.rgb[o0 + ch] * (1 - a) + nb->rgb[o1 + ch] * a) / 255.0f);
        }
    }
    b.dirty = false;
}

size_t VoxelHashTSDF::surface(std::vector<float>& out, size_t maxPoints) const {
    out.clear(); size_t n = 0;
    for (auto& kv : blocks_) { const auto& p = kv.second.pts; for (size_t i = 0; i + 5 < p.size() && n < maxPoints; i += 6, n++) out.insert(out.end(), p.begin() + i, p.begin() + i + 6); }
    return n;
}
