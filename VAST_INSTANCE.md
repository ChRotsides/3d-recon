# Vast.ai Attempt 3 H100 instance

Current paid instance:

```text
Contract:    48855489
Offer:       48393942
GPU:         1× NVIDIA H100 NVL, 95,830 MiB
CPU:         192 effective cores
RAM:         193 GB
Disk:        128 GB local container storage
Location:    Romania
Hourly cost: approximately $2.726 including storage
Created:     2026-08-27
```

Connect using the verified direct route:

```bash
ssh vast-attempt3
```

The direct endpoint is currently `109.101.128.237:40093`. The Vast proxy endpoint is recorded as
`ssh3.vast.ai:15488`, but direct SSH is currently the verified route.

The project is stored at:

```text
/workspace/3d-recon
```

The Python environment is:

```text
/workspace/attempt2-venv/bin/python
```

The workspace was copied through Vast's stopped-instance transfer from contract `48811122`.
Baseline A was verified after copying:

```text
checkpoint: 93706eb9f253f33c5a11960493005da29bf986682fbab7da7e0e179fce8bb3f8
metrics:    5d453b87caed7192366a705d28f688015e08bf06a757dee05d868fcea181ce42
```

Both old and new disks are ordinary container storage, not detachable volumes. Stopping preserves
them and continues storage billing; destroying either contract permanently deletes its disk.

The old contract `48811122` remains stopped as the source backup. Do not destroy it until the
Attempt 3 diagnostic results have been copied locally and explicitly approved.
