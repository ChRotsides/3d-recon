// Voxel-hash TSDF in plain JavaScript. No dependencies; the algorithm is the one a phone would run.
//
// Space is divided into 8x8x8-voxel blocks kept in a hash map, allocated only where depth was
// observed within the truncation band. Each voxel stores a truncated signed distance, a weight and
// a colour, updated as a running average by every depth map that sees it (Curless & Levoy).
// The surface is extracted as zero crossings between neighbouring voxels, per block, only for
// blocks touched since their last extraction.
class VoxelHashTSDF {
  constructor({ voxel = 0.02, truncVoxels = 6, maxWeight = 16, minWeight = 2, maxDepth = 6.0 } = {}) {
    this.voxel = voxel; this.trunc = truncVoxels * voxel; this.maxWeight = maxWeight; this.minWeight = minWeight; this.maxDepth = maxDepth;
    this.B = 8; this.blockSize = this.B * voxel; this.blocks = new Map(); this.touched = new Set(); this.frames = 0;
  }
  static key(bx, by, bz) { return ((bx + 32768) * 65536 + (by + 32768)) * 65536 + (bz + 32768); }
  block(bx, by, bz) {
    const k = VoxelHashTSDF.key(bx, by, bz); let b = this.blocks.get(k);
    if (!b) { const n = this.B ** 3; b = { bx, by, bz, tsdf: new Float32Array(n).fill(1), w: new Float32Array(n), rgb: new Uint8Array(n * 3), pts: null, dirty: true }; this.blocks.set(k, b); }
    return b;
  }
  // depth: Uint16Array (mm) of w*h; K = {fx,fy,cx,cy}; R = camera->world rotation (OpenCV axes, row-major 9), c = camera centre;
  // rgb: {w,h,data(RGBA)} sampled with the same normalised coordinates, or null.
  integrate(depth, w, h, K, R, c, rgb, stride = 2) {
    const t0 = performance.now(); const { fx, fy, cx, cy } = K; const T = this.trunc, V = this.voxel, BS = this.blockSize;
    const alloc = new Set(); const carveOnly = new Set(); const step = BS * 0.5; const phase = this.frames & 3;
    // 1) allocate blocks along each ray inside the truncation band
    for (let v = 0; v < h; v += stride) for (let u = 0; u < w; u += stride) {
      const z = depth[v * w + u] / 1000; if (!z || z > this.maxDepth) continue;
      if (VoxelHashTSDF.flying(depth, w, h, u, v)) continue;
      const dx = (u - cx) / fx, dy = (v - cy) / fy; const n = Math.hypot(dx, dy, 1);
      // world direction of the ray
      const rx = (R[0] * dx + R[1] * dy + R[2]) / n, ry = (R[3] * dx + R[4] * dy + R[5]) / n, rz = (R[6] * dx + R[7] * dy + R[8]) / n;
      const r0 = Math.max(0.05, (z - T) * n), r1 = (z + T) * n;
      for (let r = r0; r <= r1 + 1e-6; r += step) {
        const px = c[0] + rx * r, py = c[1] + ry * r, pz = c[2] + rz * r;
        alloc.add(VoxelHashTSDF.key(Math.floor(px / BS), Math.floor(py / BS), Math.floor(pz / BS)));
      }
      // free space between the camera and the band: existing blocks here hold stale surfaces to carve
      if (((u / stride) + (v / stride)) % 2 === 0) for (let r = 0.1; r < r0; r += BS) {
        const px = c[0] + rx * r, py = c[1] + ry * r, pz = c[2] + rz * r;
        const k = VoxelHashTSDF.key(Math.floor(px / BS), Math.floor(py / BS), Math.floor(pz / BS)); if (this.blocks.has(k)) carveOnly.add(k);
      }
    }
    // world->camera: p_c = R^T (p - c)
    let n_upd = 0; const bandKeys = new Set(alloc); for (const k of carveOnly) alloc.add(k);
    for (const k of alloc) {
      let b = this.blocks.get(k);
      if (!b) { const bz = (k % 65536) - 32768, by = (Math.floor(k / 65536) % 65536) - 32768, bx = Math.floor(k / 65536 / 65536) - 32768; b = this.block(bx, by, bz); }
      const ox = b.bx * BS + V / 2, oy = b.by * BS + V / 2, oz = b.bz * BS + V / 2; let changed = false; const sub = carveOnly.has(k) && !bandKeys.has(k);
      for (let i = 0, idx = 0; i < this.B; i++) for (let j = 0; j < this.B; j++) for (let l = 0; l < this.B; l++, idx++) {
        if (sub && ((i + j + l) & 3) !== phase) continue;
        const px = ox + i * V - c[0], py = oy + j * V - c[1], pz = oz + l * V - c[2];
        const zc = R[2] * px + R[5] * py + R[8] * pz; if (zc <= 0.05) continue;
        const xc = R[0] * px + R[3] * py + R[6] * pz, yc = R[1] * px + R[4] * py + R[7] * pz;
        const u = Math.round(fx * xc / zc + cx), v = Math.round(fy * yc / zc + cy); if (u < 0 || v < 0 || u >= w || v >= h) continue;
        const d = depth[v * w + u] / 1000; if (!d || d > this.maxDepth) continue;
        const sdf = d - zc; if (sdf < -T) continue;
        const t = sdf > T ? 1 : sdf / T; const wo = b.w[idx], wn = Math.min(this.maxWeight, wo + 1);
        b.tsdf[idx] = (b.tsdf[idx] * wo + t) / (wo + 1); b.w[idx] = wn; changed = true; n_upd++;
        if (rgb && Math.abs(sdf) < T * 0.5) {
          const ru = Math.min(rgb.w - 1, Math.round(u * rgb.w / w)), rv = Math.min(rgb.h - 1, Math.round(v * rgb.h / h)); const q = 4 * (rv * rgb.w + ru), o = idx * 3;
          b.rgb[o] = (b.rgb[o] * wo + rgb.data[q]) / (wo + 1); b.rgb[o + 1] = (b.rgb[o + 1] * wo + rgb.data[q + 1]) / (wo + 1); b.rgb[o + 2] = (b.rgb[o + 2] * wo + rgb.data[q + 2]) / (wo + 1);
        }
      }
      if (changed) { b.dirty = true; this.touched.add(k); }
    }
    this.frames++; return { ms: performance.now() - t0, blocks: alloc.size, updates: n_upd };
  }
  static flying(depth, w, h, u, v) {   // depth-edge pixel: a neighbour differs by more than 8 % of the depth
    const d = depth[v * w + u]; const th = 0.08 * d;
    if (u > 0 && Math.abs(depth[v * w + u - 1] - d) > th) return true; if (u < w - 1 && Math.abs(depth[v * w + u + 1] - d) > th) return true;
    if (v > 0 && Math.abs(depth[(v - 1) * w + u] - d) > th) return true; if (v < h - 1 && Math.abs(depth[(v + 1) * w + u] - d) > th) return true;
    return false;
  }
  voxelAt(bx, by, bz, i, j, l) {   // neighbour lookup across block borders
    if (i >= this.B) { bx++; i = 0 } if (j >= this.B) { by++; j = 0 } if (l >= this.B) { bz++; l = 0 }
    const b = this.blocks.get(VoxelHashTSDF.key(bx, by, bz)); if (!b) return null; const idx = (i * this.B + j) * this.B + l; return [b, idx];
  }
  // zero crossings along +x/+y/+z between voxels that both have enough weight -> [x,y,z,r,g,b] per point
  extractBlock(b) {
    const V = this.voxel, BS = this.blockSize, out = []; const ox = b.bx * BS + V / 2, oy = b.by * BS + V / 2, oz = b.bz * BS + V / 2;
    for (let i = 0, idx = 0; i < this.B; i++) for (let j = 0; j < this.B; j++) for (let l = 0; l < this.B; l++, idx++) {
      const w0 = b.w[idx]; if (w0 < this.minWeight) continue; const t0 = b.tsdf[idx];
      for (let ax = 0; ax < 3; ax++) {
        const nb = this.voxelAt(b.bx, b.by, b.bz, i + (ax === 0), j + (ax === 1), l + (ax === 2)); if (!nb) continue;
        const [b1, idx1] = nb; if (b1.w[idx1] < this.minWeight) continue; const t1 = b1.tsdf[idx1];
        if ((t0 < 0) === (t1 < 0) || Math.abs(t0 - t1) < 1e-6) continue;
        const a = t0 / (t0 - t1);   // fraction along the edge where tsdf == 0
        const x = ox + i * V + (ax === 0 ? a * V : 0), y = oy + j * V + (ax === 1 ? a * V : 0), z = oz + l * V + (ax === 2 ? a * V : 0);
        const o0 = idx * 3, o1 = idx1 * 3;
        out.push(x, y, z, (b.rgb[o0] * (1 - a) + b1.rgb[o1] * a) / 255, (b.rgb[o0 + 1] * (1 - a) + b1.rgb[o1 + 1] * a) / 255, (b.rgb[o0 + 2] * (1 - a) + b1.rgb[o1 + 2] * a) / 255);
      }
    }
    return out;
  }
  // refresh touched blocks (and their -x/-y/-z neighbours, whose crossings may involve them), return total point count
  extract() {
    const t0 = performance.now(); const redo = new Set(this.touched);
    for (const k of this.touched) { const b = this.blocks.get(k); for (const [dx, dy, dz] of [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]) { const kk = VoxelHashTSDF.key(b.bx + dx, b.by + dy, b.bz + dz); if (this.blocks.has(kk)) redo.add(kk); } }
    for (const k of redo) { const b = this.blocks.get(k); b.pts = this.extractBlock(b); b.dirty = false; }
    this.touched.clear(); let n = 0; for (const b of this.blocks.values()) if (b.pts) n += b.pts.length / 6;
    return { points: n, ms: performance.now() - t0, redone: redo.size };
  }
  // copy all surface points into position/colour buffers; returns count
  fill(pos, col, max) {
    let n = 0; for (const b of this.blocks.values()) { if (!b.pts) continue; const p = b.pts;
      for (let i = 0; i < p.length && n < max; i += 6, n++) { pos[3 * n] = p[i]; pos[3 * n + 1] = p[i + 1]; pos[3 * n + 2] = p[i + 2]; col[3 * n] = p[i + 3]; col[3 * n + 1] = p[i + 4]; col[3 * n + 2] = p[i + 5]; } }
    return n;
  }
  stats() { let vox = 0; for (const b of this.blocks.values()) for (let i = 0; i < b.w.length; i++) if (b.w[i] > 0) vox++; return { blocks: this.blocks.size, voxels: vox, mb: (this.blocks.size * (512 * 8 + 1536) / 1e6).toFixed(1) }; }
}
