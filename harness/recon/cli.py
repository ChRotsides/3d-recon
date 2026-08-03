"""`recon` command line — the Phase-0 harness entry point."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from recon import synth
from recon.eval.bench import BenchReport, host_info, run_scene
from recon.io.bundle import BundleDir
from recon.pipeline import Pipeline, PipelineConfig

app = typer.Typer(add_completion=False, help="Room-to-house 3D scanner harness.")
console = Console()

PRESETS = ["empty_rect", "l_shaped", "angled", "furnished", "mirror_heavy"]


def _make_depth(name: str):
    if name == "gt":
        from recon.depth.base import GroundTruthDepth

        return GroundTruthDepth()
    if name.startswith("gt+noise"):
        from recon.depth.base import GroundTruthDepth

        sigma = float(name.split(":")[1]) if ":" in name else 0.004
        est = GroundTruthDepth(noise_sigma_m=sigma)
        est.name = f"gt+noise:{sigma}"
        return est
    if name.startswith("da3"):
        from recon.depth.da3 import DA3Config, DA3Depth

        variant = name.split(":")[1] if ":" in name else "large"
        return DA3Depth(DA3Config(variant=variant))
    if name == "mapanything":
        from recon.depth.mapanything import MapAnythingDepth

        return MapAnythingDepth()
    raise typer.BadParameter(f"unknown depth model {name!r}")


def _make_masker(name: str):
    if name == "none":
        from recon.masking.base import NullMasker

        return NullMasker()
    if name == "gt":
        from recon.masking.base import GroundTruthMasker

        return GroundTruthMasker()
    if name == "sam3":
        from recon.masking.sam3 import SAM3Masker

        return SAM3Masker()
    raise typer.BadParameter(f"unknown masker {name!r}")


@app.command(name="synth")
def synth_cmd(
    out: Path = typer.Argument(..., help="Output directory for the scene(s)."),
    preset: str = typer.Option("empty_rect", help=f"One of {PRESETS} or 'all'."),
    width: int = typer.Option(320),
    height: int = typer.Option(240),
    stations: int = typer.Option(3),
    max_keyframes: int = typer.Option(150),
    vio_scale_error: float = typer.Option(1.0, help="e.g. 1.025 == 2.5%% too big."),
    drift_m: float = typer.Option(0.0),
    pose_noise_m: float = typer.Option(0.0),
    texture: float = typer.Option(1.0, help="0 == pathological white walls."),
    no_artifacts: bool = typer.Option(False, help="Disable mirror/window artifacts."),
    seed: int = typer.Option(0),
) -> None:
    """Render synthetic capture session(s) with exact ground truth."""
    names = PRESETS if preset == "all" else [preset]
    cfg = synth.CaptureConfig(
        width=width, height=height, n_stations=stations, max_keyframes=max_keyframes,
        vio_scale_error=vio_scale_error, drift_m=drift_m, pose_noise_t_m=pose_noise_m,
        texture_strength=texture, simulate_artifacts=not no_artifacts, seed=seed,
    )
    for name in names:
        target = out / name if len(names) > 1 else out
        spec = synth.preset(name)
        bundle, gt = synth.generate(spec, target, cfg)
        console.print(
            f"[green]{name}[/green]: {len(bundle.keyframes)} keyframes -> {target} "
            f"(area {gt.area_m2:.2f} m², ceiling {gt.ceiling_height_m:.2f} m)"
        )


@app.command()
def reconstruct(
    bundle: Path = typer.Argument(..., help="Session bundle directory."),
    out: Path = typer.Option(Path("out"), help="Artifact output directory."),
    depth: str = typer.Option("gt", help="gt | gt+noise:0.004 | da3:large | mapanything"),
    mask: str = typer.Option("gt", help="none | gt | sam3"),
    voxel: float = typer.Option(0.02, help="TSDF voxel size, metres."),
    no_scale: bool = typer.Option(False, help="Skip the §6.2 scale correction."),
) -> None:
    """Run the reconstruction pipeline on one session bundle."""
    cfg = PipelineConfig(apply_scale=not no_scale)
    cfg.tsdf.voxel_size_m = voxel
    bd = BundleDir.load(bundle)
    pipe = Pipeline(depth=_make_depth(depth), masker=_make_masker(mask), config=cfg)
    res = pipe.run(bd, out)

    console.print(res.scale.report())
    table = Table(title=f"room {res.room_id[:8]}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("area", f"{res.layout.area_m2:.3f} m²")
    table.add_row("perimeter", f"{res.layout.perimeter_m:.3f} m")
    table.add_row("ceiling height", f"{res.layout.ceiling_height_m:.3f} m")
    table.add_row("walls", str(len(res.layout.wall_lengths_m)))
    table.add_row(
        "wall lengths", ", ".join(f"{x:.3f}" for x in res.layout.wall_lengths_m)
    )
    table.add_row("openings", str(len(res.layout.openings)))
    table.add_row("masked fraction", f"{res.masks.masked_fraction()*100:.1f}%")
    table.add_row("runtime", f"{sum(res.timings_s.values()):.1f} s")
    console.print(table)

    gt = bd.ground_truth()
    if gt:
        from recon.eval.metrics import layout_metrics

        m = layout_metrics(res.layout, gt)
        console.print(
            f"[bold]vs ground truth[/bold]: wall err max "
            f"{m.wall_error_max_abs_pct:.2f}% ({m.wall_error_max_abs_m*100:.1f} cm), "
            f"area {m.area_error_pct:+.2f}%, ceiling {m.ceiling_error_m*100:+.1f} cm "
            f"-> {'[green]PASS[/green]' if m.passes() else '[red]FAIL[/red]'}"
        )
    console.print(f"artifacts -> {out}")


@app.command()
def bench(
    scenes: Path = typer.Argument(..., help="Directory containing scene subdirectories."),
    out: Path = typer.Option(Path("runs/bench"), help="Where to write the report."),
    depth: str = typer.Option("gt", help="Comma-separated depth models."),
    mask: str = typer.Option("gt,none", help="Comma-separated maskers."),
    baseline: Path = typer.Option(None, help="Previous report.json for the regression gate."),
    artifacts: bool = typer.Option(False, help="Write per-run artifacts too."),
) -> None:
    """Run the Phase-0 benchmark grid and emit the report table."""
    scene_dirs = sorted(p for p in scenes.iterdir() if (p / "bundle.json").exists())
    if not scene_dirs:
        if (scenes / "bundle.json").exists():
            scene_dirs = [scenes]
        else:
            raise typer.BadParameter(f"no bundles under {scenes}")

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sd in scene_dirs:
        for dname in depth.split(","):
            for mname in mask.split(","):
                console.print(f"[dim]running[/dim] {sd.name} depth={dname} mask={mname}")
                row = run_scene(
                    sd,
                    _make_depth(dname.strip()),
                    _make_masker(mname.strip()),
                    out_dir=(out / sd.name / f"{dname}_{mname}") if artifacts else None,
                )
                rows.append(row)
                status = "[green]pass[/green]" if row.passed else "[red]fail[/red]"
                detail = row.error or (
                    f"wall {row.wall_err:.2f}% area {row.area_err:.2f}%"
                )
                console.print(f"  -> {status} {detail} ({row.runtime_s:.1f}s)")

    report = BenchReport(
        rows=rows,
        host=host_info(),
        created_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )
    (out / "report.json").write_text(report.to_json())
    md = report.to_markdown()
    exit_c = report.exit_criterion()
    (out / "report.md").write_text(
        f"# Phase-0 benchmark\n\n_{report.created_utc}_\n\n"
        f"host: `{report.host}`\n\n{md}\n\n"
        f"## Phase-0 exit criterion\n\n```json\n{json.dumps(exit_c, indent=2)}\n```\n"
    )
    console.print(md)

    if exit_c["any_config_meets_exit"]:
        console.print("[bold green]Phase-0 exit criterion met[/bold green]")
    else:
        console.print("[bold red]Phase-0 exit criterion NOT met[/bold red]")

    if baseline and baseline.exists():
        check = report.regression_check(BenchReport.from_json(baseline.read_text()))
        if check["passed"]:
            console.print("[green]no wall-length regression[/green]")
        else:
            console.print(f"[bold red]REGRESSION[/bold red] {json.dumps(check['failures'])}")
            raise typer.Exit(code=1)

    console.print(f"report -> {out/'report.md'}")


@app.command()
def validate(
    bundle: Path = typer.Argument(..., help="Session bundle directory from a client."),
) -> None:
    """Check a client-produced bundle before it is used for data collection."""
    from recon.io.validate import validate_bundle

    rep = validate_bundle(bundle)

    table = Table(title="bundle", show_header=False)
    for k, v in rep.stats.items():
        table.add_row(k, str(v))
    console.print(table)

    for w in rep.warnings:
        console.print(f"[yellow]warn[/yellow]  {w}")
    for e in rep.errors:
        console.print(f"[red]error[/red] {e}")

    if rep.ok:
        console.print(
            "[green]OK[/green] — bundle is usable"
            + (f" ({len(rep.warnings)} warnings)" if rep.warnings else "")
        )
    else:
        console.print(f"[bold red]{len(rep.errors)} errors[/bold red] — do not collect with this")
        raise typer.Exit(code=1)


@app.command()
def viewer(
    rooms: Path = typer.Argument(..., help="Directory of per-room artifact folders."),
    out: Path = typer.Option(Path("scene.json"), help="Where to write the manifest."),
    property_id: str = typer.Option("property"),
) -> None:
    """Build a walkthrough scene manifest from reconstructed rooms (§3.8).

    Reads each room's `layout.json` and lists whatever artifacts are present, so
    it works whether or not the GPU splat stage has run.
    """
    from recon.io.viewer import build_scene, write_scene
    from recon.schemas import RoomLayout

    entries = {}
    for d in sorted(p for p in rooms.iterdir() if (p / "layout.json").exists()):
        layout = RoomLayout.model_validate_json((d / "layout.json").read_text())
        artifacts = {
            kind: f"{d.name}/{name}"
            for kind, name in (("spz", "room.spz"), ("mesh", "room.glb"),
                               ("plan_svg", "floorplan.svg"), ("plan_pdf", "floorplan.pdf"))
            if (d / name).exists()
        }
        entries[layout.room_id] = {"layout": layout, "artifacts": artifacts}

    if not entries:
        raise typer.BadParameter(f"no reconstructed rooms under {rooms}")
    scene = build_scene(property_id, entries)
    write_scene(out, scene)
    console.print(
        f"[green]{len(scene.rooms)} room(s)[/green], {scene.total_area_m2:.2f} m² -> {out}"
    )
    console.print("open viewer/index.html?scene=<path to this file>")


@app.command()
def scale(
    bundle: Path = typer.Argument(...),
) -> None:
    """Solve room scale from the bundle's reference measurements (§6.2)."""
    from recon.metric.scale import build_constraints, solve_scale

    bd = BundleDir.load(bundle)
    sol = solve_scale(build_constraints(bd.bundle.reference_measurements))
    console.print(sol.report())


if __name__ == "__main__":
    app()
