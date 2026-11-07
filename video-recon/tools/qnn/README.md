# UniDepth on the Hexagon NPU (QNN)

Prepares the original-quality UniDepth v2 ViT-S depth model (182x322 input, unchanged internal
geometry) for `QnnDepthSession` in the live scanner. Nothing here downloads anything or overwrites
existing files.

## Steps

1. Export the fp32 source (needs the UniDepth package and weights, obtained separately):
   `python vio/export_unidepth.py 182 322 unidepth_vits_182x322.onnx`
2. Prepare the two app models:
   `python -I tools/qnn/prepare_unidepth_qnn.py --source unidepth_vits_182x322.onnx --out-dir OUT`
3. Check CPU equivalence on real phone captures (depth-evidence folders):
   `python -I tools/qnn/verify_unidepth_qnn.py --source unidepth_vits_182x322.onnx --prepared-dir OUT --evidence-dir CAPTURES`
4. Copy `OUT/qnn_camera.onnx` and `OUT/qnn_main_static_gelu.onnx` into the app's `files/models/`.

`prepare_unidepth_qnn.py` folds constants, pins every shape and splits the model at the tensors
that depend only on K:

- `qnn_camera.onnx`: K [1,3,3] to nine camera-conditioning tensors. Runs on the CPU for every frame,
  so intrinsics stay dynamic.
- `qnn_main_static_gelu.onnx`: image [1,3,182,322] plus those tensors to depth [1,182,322]. All
  shapes are static. The exact-GELU `Erf` expansions are fused into ONNX `Gelu` (opset 20), because
  the ONNX Runtime 1.22 QNN execution provider has no `Erf` builder. The script fails if any op
  lacks a QNN 1.22 builder or any tensor shape is dynamic.

It also writes `qnn_prepare_manifest.json` with SHA-256 hashes and tool versions.

## App contract (`QnnDepthSession.kt`)

- Needs `com.microsoft.onnxruntime:onnxruntime-android-qnn` (1.22.0 tested; it pulls in
  `com.qualcomm.qti:qnn-runtime`). With plain `onnxruntime-android` the constructor throws
  `UnsupportedOperationException`. The class still compiles there, because `addQnn` is called by
  reflection.
- `run(chw, k)`: `chw` is ImageNet-normalised RGB, 3x182x322, in the same layout as `DepthRunner`.
  `k` is fx, fy, cx, cy at 322x182. Returns metric depth, 182x322.
- The main graph uses QNN HTP with `session.disable_cpu_ep_fallback=1` and FP16 precision
  (`enable_htp_fp16_precision=1`, `htp_performance_mode=burst`). `htp_arch` is omitted unless
  the caller passes one.
- `ADSP_LIBRARY_PATH` is set to the app's native library directory, then any existing value, then
  the standard vendor DSP directories.

### Context cache

Compiling the main graph took 203.7 s on SM-S918B, so the result is cached. Layout under the
caller's cache directory:

```
qnn-depth-context-v1/
  .lock                      cross-process build lock (FileChannel.lock)
  <key>/main_ctx.onnx        EP context model, embedded QNN binary (ep.context_embed_mode=1)
  <key>/manifest.properties  identity, context size and SHA-256, compile time
  .build-<key>-<uuid>/       staging during a compile; renamed to <key>/ in one atomic move
  quarantine/<key>-<millis>/ entries that failed validation (newest two kept)
```

`<key>` is the first 32 hex digits of a SHA-256 over:

- the main model's SHA-256 and size;
- the QNN provider options and session settings;
- the ONNX Runtime version;
- the SHA-256 of every `libonnxruntime*`, `libQnnHtp*` and `libQnnSystem*` library in the native
  library directory;
- the device (manufacturer, model, hardware, board, SoC, build fingerprint, SDK level).

An entry loads without compiling only if its manifest identity, context size and context SHA-256
all match. Otherwise it is moved to `quarantine/` (never deleted in place) and rebuilt. A fresh
compile writes into `.build-*` with `ep.context_enable=1` and `ep.context_file_path`, then
publishes. The compilation session closes before reopening the published context, so two
large NPU sessions do not coexist. Intact cached contexts that fail to load trigger
CPU fallback; transient DSP errors do not cause automatic recompilation.

Pass a directory the system does not purge, for example `Context.getNoBackupFilesDir()`. An entry
holds the compiled model.

## Measured

- CPU equivalence of camera + main against the source on 41 phone captures, each with its own K:
  max |diff| 1.9e-6 m, max relative 7.2e-7.
- SM-S918B with QNN 1.22.0 and `session.disable_cpu_ep_fallback=1`:
  - camera graph on CPU: 21-31 ms;
  - main graph on HTP: about 220 ms;
  - first compile: 203.7 s.

## Licences

Model weights are not bundled. The UniDepth weights and the Qualcomm QNN runtime are distributed
under their own licences: the `qnn-runtime` Maven POM declares the "Qualcomm AI Hub Model License".
Review both before shipping.
