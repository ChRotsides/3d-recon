// Voxel-hash TSDF, the C++ twin of video-recon/vio/tsdf.js: 8^3 blocks in a hash map allocated
// along rays inside the truncation band, projective integration with a weighted running average of
// distance and colour, free-space carving along rays, zero-crossing surface extraction per touched block.
#pragma once
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>

struct TsdfBlock {
    int bx, by, bz;
    float tsdf[512];
    float w[512];
    uint8_t rgb[512 * 3];
    std::vector<float> pts;   // extracted surface: x,y,z,r,g,b per point (rgb 0..1)
    std::vector<float> confidence;
    bool dirty = true;
};

class VoxelHashTSDF {
public:
    explicit VoxelHashTSDF(float voxel = 0.03f, int truncVoxels = 4, float maxWeight = 16, float minWeight = 2, float maxDepth = 6, size_t maxBlocks = 40000);
    // status: 0 integrated, 1 memory budget reached, 2 invalid input. A rejected
    // frame makes no changes, so the current scan can still be saved in full.
    struct Stats { double integrate_ms = 0, extract_ms = 0; int blocks_touched = 0, blocks = 0; size_t points = 0; int status = 0; };
    // depth: mm, row-major w*h; K: fx,fy,cx,cy; R: camera->world (OpenCV axes) row-major 3x3; c: camera centre;
    // rgb: optional RGBA (rw*rh*4) sampled with normalised coordinates.
    Stats integrate(const uint16_t* depth, int w, int h, const float K[4], const float R[9], const float c[3],
                    const uint8_t* rgb, int rw, int rh, int stride, const uint8_t* confidence = nullptr);
    size_t surface(std::vector<float>& out, size_t maxPoints) const;
    int64_t saveSurface(const char* path) const;
    size_t blocks() const { return blocks_.size(); }
    bool hasSurface() const {
        for (const auto& entry : blocks_) if (!entry.second.pts.empty()) return true;
        return false;
    }
    void clear() { blocks_.clear(); touched_.clear(); frames_ = 0; }
private:
    static inline int64_t key(int bx, int by, int bz) { return (((int64_t)(bx + 32768) * 65536 + (by + 32768)) * 65536) + (bz + 32768); }
    static inline void unkey(int64_t k, int& bx, int& by, int& bz) { bz = (int)(k % 65536) - 32768; k /= 65536; by = (int)(k % 65536) - 32768; k /= 65536; bx = (int)k - 32768; }
    TsdfBlock& block(int64_t k, int bx, int by, int bz);
    static bool flying(const uint16_t* d, int w, int h, int u, int v);
    void extractBlock(TsdfBlock& b);
    const TsdfBlock* neighbour(int bx, int by, int bz, int& i, int& j, int& l, int& idx) const;
    float voxel_, trunc_, maxW_, minW_, maxDepth_, blockSize_;
    std::unordered_map<int64_t, TsdfBlock> blocks_;
    std::unordered_set<int64_t> touched_;
    int frames_ = 0;
    size_t maxBlocks_;
};
