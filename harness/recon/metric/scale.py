"""Metric scale solver (PLAN.md §6.2).

VIO already provides metric scale from the IMU. The reference measurement
**corrects slow drift; it does not create scale.** That framing is load-bearing:
the VIO prior is a first-class constraint that is never discarded, so a bad tape
entry degrades the answer instead of destroying it.

    s* = argmin  sum_i (1/sigma_i^2) * (s * d_i_est - d_i_obs)^2

which is a scalar weighted least squares with a closed-form solution:

    s* = sum_i w_i * d_est_i * d_obs_i / sum_i w_i * d_est_i^2 ,  w_i = 1/sigma_i^2

Scale priors (VIO, MapAnything) are entered as d_est = 1, d_obs = prior, so they
sit in the same normal equations as the distance constraints rather than being
bolted on afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recon.schemas import MeasurementType, ReferenceMeasurement

#: Sigmas from the PLAN.md §6.2 table, used when a measurement omits its own.
DEFAULT_SIGMA: dict[MeasurementType, float] = {
    MeasurementType.VIO_SCALE: 0.03,
    MeasurementType.USER_TAPE: 0.005,
    MeasurementType.DOOR_PRIOR: 0.04,
    MeasurementType.CEILING_PRIOR: 0.10,
    MeasurementType.CAMERA_HEIGHT: 0.08,
    MeasurementType.MAPANYTHING_SCALE: 0.10,  # TBD until Phase 0 measures it
}

#: Scale priors are unitless; distance constraints are in metres.
SCALE_PRIOR_TYPES = frozenset({MeasurementType.VIO_SCALE, MeasurementType.MAPANYTHING_SCALE})


@dataclass
class ScaleConstraint:
    """One row of the normal equations."""

    kind: MeasurementType
    d_est: float  # distance in the reconstruction (1.0 for a scale prior)
    d_obs: float  # the real-world truth
    sigma: float
    label: str = ""

    @property
    def weight(self) -> float:
        return 1.0 / (self.sigma * self.sigma)

    def residual(self, s: float) -> float:
        return s * self.d_est - self.d_obs

    def normalised_residual(self, s: float) -> float:
        return self.residual(s) / self.sigma


@dataclass
class ScaleSolution:
    scale: float
    #: 1-sigma uncertainty on `scale` from the normal equations.
    sigma: float
    constraints: list[ScaleConstraint]
    rejected: list[ScaleConstraint]
    iterations: int

    @property
    def correction_pct(self) -> float:
        return (self.scale - 1.0) * 100.0

    def report(self) -> str:
        lines = [
            f"scale = {self.scale:.6f} +/- {self.sigma:.6f} "
            f"({self.correction_pct:+.2f}% correction)",
        ]
        for c in self.constraints:
            lines.append(
                f"  [used]     {c.kind.value:<20s} est={c.d_est:8.4f} obs={c.d_obs:8.4f} "
                f"sigma={c.sigma:.4f} r/sigma={c.normalised_residual(self.scale):+7.2f} {c.label}"
            )
        for c in self.rejected:
            lines.append(
                f"  [REJECTED] {c.kind.value:<20s} est={c.d_est:8.4f} obs={c.d_obs:8.4f} "
                f"sigma={c.sigma:.4f} r/sigma={c.normalised_residual(self.scale):+7.2f} {c.label}"
            )
        return "\n".join(lines)


def build_constraints(
    measurements: list[ReferenceMeasurement],
    *,
    mapanything_scale: float | None = None,
    mapanything_sigma: float | None = None,
) -> list[ScaleConstraint]:
    """Turn bundle measurements into solver rows.

    A distance constraint's `d_est` is the length of the segment *as measured in
    the reconstruction*; `d_obs` is the tape reading. A scale prior contributes
    d_est = 1.
    """
    out: list[ScaleConstraint] = []
    for m in measurements:
        sigma = m.sigma_m or DEFAULT_SIGMA.get(m.type, 0.05)
        if m.type in SCALE_PRIOR_TYPES:
            out.append(ScaleConstraint(m.type, 1.0, m.measured_value_m, sigma, "prior"))
            continue
        if m.point_a is None or m.point_b is None:
            continue
        d_est = float(np.linalg.norm(np.asarray(m.point_b) - np.asarray(m.point_a)))
        if d_est < 1e-6:
            continue
        # A distance constraint's sigma is on the *observed* length, so it maps
        # to the same units as the residual without rescaling.
        out.append(ScaleConstraint(m.type, d_est, m.measured_value_m, sigma))

    if mapanything_scale is not None:
        out.append(
            ScaleConstraint(
                MeasurementType.MAPANYTHING_SCALE,
                1.0,
                mapanything_scale,
                mapanything_sigma or DEFAULT_SIGMA[MeasurementType.MAPANYTHING_SCALE],
                "scale token",
            )
        )
    return out


def solve_scale(
    constraints: list[ScaleConstraint],
    *,
    huber_delta: float = 2.5,
    max_iterations: int = 12,
    reject_at: float = 6.0,
) -> ScaleSolution:
    """Weighted least squares with Huber IRLS and hard outlier rejection.

    A user who types 83 when they meant 0.83 must not be able to shrink the
    house. Huber bounds the influence of a wrong entry; anything past
    `reject_at` sigma after convergence is dropped and the solve repeated.
    """
    if not constraints:
        raise ValueError("no scale constraints; the VIO prior must always be present")

    active = list(constraints)
    rejected: list[ScaleConstraint] = []
    s = 1.0
    iterations = 0

    for _outer in range(3):
        for _ in range(max_iterations):
            iterations += 1
            num = den = 0.0
            for c in active:
                r = c.normalised_residual(s)
                # Huber: quadratic inside delta, linear outside.
                rho = 1.0 if abs(r) <= huber_delta else huber_delta / abs(r)
                w = c.weight * rho
                num += w * c.d_est * c.d_obs
                den += w * c.d_est * c.d_est
            if den <= 0:
                raise ValueError("degenerate scale system (all estimated distances are zero)")
            s_new = num / den
            if abs(s_new - s) < 1e-12:
                s = s_new
                break
            s = s_new

        outliers = [c for c in active if abs(c.normalised_residual(s)) > reject_at]
        # Never reject the VIO prior — it is the anchor that keeps a room with
        # one bad tape entry from collapsing onto that entry.
        outliers = [c for c in outliers if c.kind is not MeasurementType.VIO_SCALE]
        if not outliers or len(active) - len(outliers) < 1:
            break
        rejected.extend(outliers)
        active = [c for c in active if c not in outliers]

    den = sum(
        c.weight
        * (1.0 if abs(c.normalised_residual(s)) <= huber_delta
           else huber_delta / abs(c.normalised_residual(s)))
        * c.d_est**2
        for c in active
    )
    sigma = float(np.sqrt(1.0 / den)) if den > 0 else float("inf")

    return ScaleSolution(
        scale=float(s), sigma=sigma, constraints=active, rejected=rejected, iterations=iterations
    )
