// Runs the TSDF off the main thread. Messages in: {type:'reset', opts} | {type:'integrate', ...};
// out: {type:'surface', pos, col, n, stats} with transferred buffers.
importScripts('tsdf.js');
let tsdf = null, opts = {};
onmessage = e => {
  const m = e.data;
  if (m.type === 'reset') { opts = m.opts; tsdf = new VoxelHashTSDF(opts); postMessage({ type: 'ready' }); return; }
  if (m.type !== 'integrate' || !tsdf) return;
  const depth = new Uint16Array(m.depth); const rgb = m.rgb ? { w: m.rgbw, h: m.rgbh, data: new Uint8ClampedArray(m.rgb) } : null;
  const r = tsdf.integrate(depth, m.w, m.h, m.K, m.R, m.c, rgb, m.stride);
  const x = tsdf.extract(); const max = m.max || 4000000;
  const pos = new Float32Array(Math.min(max, x.points) * 3), col = new Float32Array(pos.length);
  const n = tsdf.fill(pos, col, max); if (tsdf.frames % 5 === 1 || !onmessage.st) onmessage.st = tsdf.stats(); const st = onmessage.st;
  postMessage({ type: 'surface', pos: pos.buffer, col: col.buffer, n, stats: { integrate_ms: r.ms, blocks_touched: r.blocks, extract_ms: x.ms, blocks: st.blocks, voxels: st.voxels, mb: st.mb } }, [pos.buffer, col.buffer]);
};
