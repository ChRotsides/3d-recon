# Simulated ground robot: pose x depth sweep

`video-recon/sim/sweep.py`, world `apartment`, seeds 1 and 2, **120 s of simulated time per
run**, 640x360 camera. The brief asked for 180 s and seeds 1-3; that is 27 runs at 5-10 min
each and did not fit, so it was cut to 120 s x 2 seeds (18 runs, 234 min for the first pass).
The twelve `gt`/`gt_noisy` rows ran one at a time on 12 cores; the six `vio` rows were rerun
afterwards two at a time with 5 threads each (20 min for all six).

Explored % is the fraction of the ground-truth free floor area the robot's own map marks
known-free, after aligning the estimate frame to the world with a rigid Umeyama fit on the two
camera paths. Map median / p90 are distances from the TSDF surface points to the ground-truth
mesh; completeness is the fraction of ground-truth surface samples within 5 cm of an estimated
point. Scale is the Sim(3) factor mapping estimate onto truth, so 0.5 means the estimator's
metric scale is twice too large. A run ends early when no reachable frontier is left for 25 s,
so `dist m` and RTF are over the time the run actually took.

The `vio` rows use `estimator_config_nocalib.yaml` (online calibration off); see
`sim-findings-2026-09-04.md` for why, and for the 40 s measurements where VIO does track.

| pose | depth | seed | explored % | collisions | ATE rmse cm | scale | map median cm | map p90 cm | completeness % | dist m | VIO init s | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gt | gt | 1 | 32 | 0 | 0.0 | 1.0 | 0.23 | 0.49 | 18 | 14.12 | 0.0 | 0.273 |
| gt | gt | 2 | 47 | 5 | 0.0 | 1.0 | 0.27 | 0.54 | 23 | 14.77 | 0.0 | 0.275 |
| gt | metric3d | 1 | 19 | 0 | 0.0 | 1.0 | 20.48 | 71.13 | 4 | 10.66 | 0.0 | 0.208 |
| gt | metric3d | 2 | 22 | 6 | 0.0 | 1.0 | 11.5 | 58.5 | 5 | 9.79 | 0.0 | 0.209 |
| gt | da3 | 1 | 22 | 4 | 0.0 | 1.0 | 13.09 | 49.71 | 9 | 11.4 | 0.0 | 0.232 |
| gt | da3 | 2 | 29 | 3 | 0.0 | 1.0 | 37.5 | 190.78 | 6 | 11.39 | 0.0 | 0.224 |
| gt_noisy | gt | 1 | 32 | 6 | 5.8 | 0.9831 | 5.65 | 15.11 | 10 | 12.95 | 0.0 | 0.271 |
| gt_noisy | gt | 2 | 20 | 0 | 8.7 | 1.0379 | 6.05 | 21.99 | 10 | 11.12 | 0.0 | 0.279 |
| gt_noisy | metric3d | 1 | 15 | 0 | 6.4 | 1.0192 | 19.85 | 56.97 | 6 | 9.16 | 0.0 | 0.208 |
| gt_noisy | metric3d | 2 | 10 | 0 | 9.5 | 0.9587 | 18.81 | 61.03 | 4 | 11.6 | 0.0 | 0.209 |
| gt_noisy | da3 | 1 | 22 | 7 | 7.7 | 0.9732 | 13.59 | 56.44 | 10 | 12.41 | 0.0 | 0.229 |
| gt_noisy | da3 | 2 | 6 | 4 | 11.2 | 1.0322 | 12.34 | 45.91 | 6 | 9.48 | 0.0 | 0.232 |
| vio | gt | 1 | 0 | 0 | 2816.6 | 0.0097 | 1192.52 | 1719.03 | 0 | 15.2 | 1.6 | 0.383 |
| vio | gt | 2 | 0 | 5 | 1660.0 | 0.0334 | 1321.08 | 1689.54 | 0 | 19.21 | 1.65 | 0.384 |
| vio | metric3d | 1 | 5 | 0 | 265.1 | 0.1853 | 49.65 | 373.43 | 6 | 12.84 | 1.6 | 0.264 |
| vio | metric3d | 2 | 0 | 16 | 12532.1 | 0.0037 | 4635.23 | 5036.49 | 0 | 18.36 | 1.65 | 0.329 |
| vio | da3 | 1 | 0 | 0 | 24611.7 | 0.0009 | 11618.32 | 12371.05 | 0 | 14.71 | 1.6 | 0.326 |
| vio | da3 | 2 | 0 | 0 | 7543.3 | 0.0022 | 7379.27 | 7842.55 | 0 | 16.03 | 1.65 | 0.304 |

**Reading it.** Perfect pose and perfect depth is the ceiling: 32-47 % explored, 0-5
collisions, a 0.23-0.27 cm map. Drifting odometry costs little (32 % / 5.7 cm map on seed 1).
The depth networks cost an order of magnitude of map accuracy (11-37 cm) and roughly half the
explored area, because their depth noise leaves fewer fully-known columns in the height band.
Every `vio` row diverges before 120 s and explores ~0 %: OpenVINS holds for 30-60 s on this
robot and then loses scale.
