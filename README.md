# Local Model Serving

Lightweight local serving prototype for ONNX models on resource-constrained
devices. The service keeps model versions on disk, runs inference in isolated
worker processes, exposes model management endpoints, and records timing and
resource metrics.

The repository stores only source code, experiment scripts, and small execution
metadata. Model files, generated tensors, logs, and experiment outputs are
ignored by Git.

## Project Layout

```text
app/                         serving application
client/                      CLI client for model management
config/                      local active-model configuration
experiments/measurements/    baseline and serving experiment runners
experiments/support/         shared experiment utilities
experiments/*/*.json         model input/output metadata
scripts/                     service and experiment entry points
```

## Setup

Python 3.11+ is expected.

```bash
cd model-serving
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Download the model artifacts used by the experiments:

```bash
bash scripts/download_artifacts.sh
```

This creates local ONNX files under `experiments/*/artifacts/`. These files are
runtime inputs only and are not committed.

## Run the Service

```bash
bash scripts/prepare_active_model.sh
bash scripts/start_service.sh
curl http://127.0.0.1:8080/ready
curl http://127.0.0.1:8080/v1/models
bash scripts/stop_service.sh
```

## Run Experiments

All reports are written to `experiments/results/<run-id>/`.

### Single-model comparison

Compares direct ONNX Runtime inference with the serving system for one YOLO26n
model.

```bash
bash scripts/run_single_experiment.sh latency
bash scripts/run_single_experiment.sh resources
bash scripts/run_single_experiment.sh diagnostics
bash scripts/run_single_experiment.sh update
bash scripts/run_single_experiment.sh rollback
bash scripts/run_single_experiment.sh recovery
```

Main report fields: `result.baseline` and `result.proposed`.

### Two versions of one model

Measures concurrent serving for two active versions of YOLO26n.

```bash
bash scripts/run_two_model_experiment.sh
```

### Two different models

Measures concurrent serving and baseline behavior for YOLO26n and
EfficientNet-Lite4.

```bash
bash scripts/run_mixed_concurrent_experiment.sh
```

Main output files: `mixed_concurrent.json`, `baseline.json`, `serving.json`,
and `summary.json`.

## Experiment Notes

Baseline details: `experiments/measurements/baseline/README.md`.
Serving details: `experiments/measurements/serving/README.md`.
