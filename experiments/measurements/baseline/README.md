# Baseline Experiments

The baseline runs ONNX Runtime directly, without the serving layer. It is used
as the reference point for latency, resource usage, and recovery scenarios.

## Scripts

```text
experiments/measurements/baseline_direct.py
experiments/measurements/baseline_daemon.py
```

## Run

The baseline is executed together with the serving variant:

```bash
bash scripts/run_single_experiment.sh latency
bash scripts/run_single_experiment.sh resources
bash scripts/run_single_experiment.sh update
bash scripts/run_single_experiment.sh rollback
bash scripts/run_single_experiment.sh recovery
```

Results are stored in `result.baseline` inside the generated JSON report.
