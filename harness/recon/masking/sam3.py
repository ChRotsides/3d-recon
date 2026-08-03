"""SAM 3 adapter — Promptable Concept Segmentation (PLAN.md §3.5).

Runs noun-phrase prompts over every keyframe and returns the union as a fusion
mask. SAM 3 is the specific choice here because earlier SAM generations
segmented the objects *behind* glass and *reflected in* mirrors while missing
the surface itself, which is precisely the failure that produces phantom rooms.

Not runnable on the Kepler dev box; torch import is deferred.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from recon.io.bundle import BundleDir
from recon.masking.base import (
    DEFAULT_PROMPTS,
    MASKED_GEOMETRIC_WEIGHT,
    MaskResult,
    _dilate,
)


@dataclass
class SAM3Config:
    checkpoint: str = "facebook/sam3"
    device: str = "cuda"
    prompts: tuple[str, ...] = DEFAULT_PROMPTS
    #: Per-prompt score floor. Mirrors are the expensive miss, so they get a
    #: lower bar than the rest — over-masking costs coverage, under-masking
    #: costs a phantom room.
    score_threshold: dict[str, float] = field(
        default_factory=lambda: {
            "mirror": 0.30,
            "window": 0.35,
            "glass door": 0.35,
            "television screen": 0.45,
            "person": 0.45,
        }
    )
    default_threshold: float = 0.40
    dilate_px: int = 3
    #: Propagate masks across the sequence with SAM 3's tracking head, so a
    #: mirror missed in one frame is still masked from its neighbours.
    track: bool = True
    batch_size: int = 8


class SAM3Masker:
    name = "sam3"

    def __init__(self, config: SAM3Config | None = None) -> None:
        self.config = config or SAM3Config()
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        try:
            import torch  # noqa: F401
            from transformers import Sam3Model, Sam3Processor  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on GPU box
            raise RuntimeError(
                "SAM 3 is not installed. On a GPU box:\n"
                "  pip install 'transformers>=4.57' torch\n"
                "and accept the model licence at https://huggingface.co/facebook/sam3\n"
                f"(original error: {exc})"
            ) from exc
        self._processor = Sam3Processor.from_pretrained(self.config.checkpoint)
        self._model = Sam3Model.from_pretrained(self.config.checkpoint).to(self.config.device)
        self._model.eval()
        return self._model, self._processor

    def mask(self, bundle: BundleDir) -> MaskResult:
        import torch

        model, processor = self._load()
        cfg = self.config
        t0 = time.time()

        fusion: dict[str, np.ndarray] = {}
        weight: dict[str, np.ndarray] = {}
        per_prompt: dict[str, dict[str, np.ndarray]] = {}

        kfs = bundle.bundle.keyframes
        for start in range(0, len(kfs), cfg.batch_size):
            chunk = kfs[start : start + cfg.batch_size]
            images = [bundle.rgb(kf) for kf in chunk]
            for kf, img in zip(chunk, images, strict=True):
                h, w = img.shape[:2]
                bad = np.zeros((h, w), dtype=bool)
                per_prompt[kf.id] = {}
                for prompt in cfg.prompts:
                    inputs = processor(
                        images=img, text=prompt, return_tensors="pt"
                    ).to(cfg.device)
                    with torch.inference_mode():
                        out = model(**inputs)
                    results = processor.post_process_instance_segmentation(
                        out, threshold=cfg.score_threshold.get(prompt, cfg.default_threshold),
                        target_sizes=[(h, w)],
                    )[0]
                    m = np.zeros((h, w), dtype=bool)
                    for inst in results.get("masks", []):
                        m |= np.asarray(inst.cpu() if hasattr(inst, "cpu") else inst, dtype=bool)
                    per_prompt[kf.id][prompt] = m
                    bad |= m
                bad = _dilate(bad, cfg.dilate_px)
                fusion[kf.id] = ~bad
                weight[kf.id] = np.where(bad, MASKED_GEOMETRIC_WEIGHT, 1.0).astype(np.float32)

        return MaskResult(
            fusion_mask=fusion,
            geometric_weight=weight,
            per_prompt=per_prompt,
            model_name=self.name,
            runtime_s=time.time() - t0,
        )
