import json, sys
from pathlib import Path
recs = [("rec0045", "rec_20260903-070045: first walk, 67 s, no still start, no loop", "plots.html"),
        ("rec2445", "rec_20260903-232445: protocol walk, 77 s, still start + bounces + loop back", None)]
rows = ""
for key, title, plots in recs:
    d = json.loads(Path(key, "walk.json").read_text())
    hdr = "<tr><th>run</th><th>first pose</th><th>path</th><th>extent (m)</th><th>start→end drift</th>" + ("<th>vs COLMAP median / rmse</th><th>scale fitted</th>" if d["ref"] else "") + "</tr>"
    trs = ""
    for n, nt in d["notes"].items():
        trs += f"<tr><td>{n}</td><td>{nt['first_pose_s']} s</td><td>{nt['path_m']} m</td><td>{' × '.join(str(x) for x in nt['extent_m'])}</td><td>{nt['start_end_drift_cm']:.0f} cm</td>" + (f"<td>{nt['vs_base_median_cm']} / {nt['vs_base_rmse_cm']} cm</td><td>{nt['scale_applied']}</td>" if d["ref"] else "") + "</tr>"
    links = f'<a href="walk2.html?rec={key}">video + 3D walk</a>' + (f' &nbsp; <a href="{plots}">trajectory plots vs COLMAP</a>' if plots else "") + (" &nbsp; <span style='color:#888'>COLMAP reference: " + ("available" if d["ref"] else "running, check back") + "</span>")
    rows += f"<section><h2>{title}</h2><p>{links}</p><table>{hdr}{trs}</table></section>"
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>VIO walks</title><style>body{{background:#111;color:#ddd;font-family:system-ui;max-width:1100px;margin:24px auto;padding:0 16px}}
a{{color:#8cf}} table{{border-collapse:collapse;font-size:14px}} td,th{{border:1px solid #333;padding:4px 10px;text-align:left}} th{{background:#1c1c1c}} h2{{font-size:17px;margin-top:28px}} p.n{{color:#999;font-size:13px}}</style></head><body>
<h1 style="font-size:20px">VIO on the phone recordings</h1>
<p class="n">Drift is the metric distance between the first VIO pose and the average pose over the last second, no alignment. "vs COLMAP" is after a Sim(3) fit to the COLMAP + floor-scale reference, so it measures shape, not scale; "scale fitted" is how much VIO had to be shrunk to match (1.0 would mean VIO's metric scale agrees with the reference).</p>
{rows}
<p class="n">OpenVINS, mono, 960×540, 250 features, online calibration of extrinsics, intrinsics and time offset. 9 ms per frame on the desktop.</p>
</body></html>"""
Path("index.html").write_text(html); print("index written")
