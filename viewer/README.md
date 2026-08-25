# Viewers

Two dependency-free single files. No build step, no CDN, no vendored library —
a capture review happens inside the property being scanned, and a viewer that
needs signal in someone's basement is not a viewer.

| File | Reads | Shows |
|---|---|---|
| `panorama.html` | `sessions/manifest.json` | 360 equirectangular panoramas, canvas pan/zoom |
| `cloud.html` | `cloud.bin` + `cloud.json` | WebGL point cloud, orbit/zoom/pan, texture toggle |

## Use

Serve them beside the artefacts they read — the paths are relative:

```bash
python scripts/capture/equirect.py data/captures/bundle_<uuid> out/sessions/equirect.jpg 4096
python scripts/capture/da3_cloud.py data/captures/bundle_<uuid> out/cloud.bin small 900000

cp viewer/panorama.html out/index.html
cp viewer/cloud.html    out/cloud.html
cd out && python -m http.server 8777 --bind 0.0.0.0
```

## Two things that are easy to get wrong

**Seam wrapping is declared, not guessed.** `manifest.json` carries
`"wrap": true` per source. Inferring it from aspect ratio fails as soon as the
poles are cropped: a full 360 came out 2.58:1 and would have missed a 2.6
threshold.

**Point size is a physical radius**, `radius_metres × focal_px / depth`, not an
arbitrary pixel constant. The first version used `uSize * 300.0 / w`, which made
every point ~62 px across at a 12 m view — a million overlapping blobs read as a
solid wall of colour rather than as a room.

The cloud is one flat buffer — `float32 xyz` for every point, then `uint8 rgb` —
straight into GPU buffers, because at ~900k points a PLY text parse costs more
than the download.
