# Walkthrough viewer

Plan §1.1 ("photorealistic walkthrough of the whole property") and §3.8, which
chooses an embedded WebView over writing and maintaining two native splat
renderers.

## Use

```bash
recon reconstruct data/scenes/l_shaped --out runs/room   # writes scene.json
recon viewer runs/ --out runs/scene.json                 # or a whole property
python -m http.server -d . 8080
# open http://localhost:8080/viewer/index.html?scene=/runs/scene.json
```

## Vendoring Babylon

`viewer/vendor/babylon.js` and `babylon.loaders.js` are **not committed** — they
are build-time assets the mobile app bundles into its WebView. Fetch them for
local use:

```bash
mkdir -p viewer/vendor
curl -Lo viewer/vendor/babylon.js         https://cdn.babylonjs.com/babylon.js
curl -Lo viewer/vendor/babylon.loaders.js https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js
```

Loaded locally rather than from a CDN on purpose: a capture review happens
inside the property being scanned, and a viewer that needs signal in someone's
basement is not a viewer. The page reports a clear error if the files are
missing instead of rendering a blank canvas.

## The frame

Babylon.js is **left-handed** by default; everything upstream here is
right-handed, Y-up (plan §5). The scene manifest carries both its frame and the
conversion required, and `applyFrame()` **throws** rather than defaulting when a
scene does not say. Guessing wrong renders a mirrored house that looks entirely
plausible — plan §7's failure mode, and one that only a measurement catches.
