# Serving Experiments

The serving variant starts the local API in an isolated run directory, prepares
model versions, sends tensor requests, collects metrics, and stops the service.

## Runtime Paths

```text
SERVING_MODEL_ROOT=<run-dir>/proposed/models
SERVING_CONFIG_PATH=<run-dir>/proposed/config/active_models.json
SERVING_OBSERVABILITY_RING_PATH=<run-dir>/proposed/logs/observability.ring
```

## Run

```bash
bash scripts/run_single_experiment.sh latency
bash scripts/run_single_experiment.sh resources
bash scripts/run_single_experiment.sh diagnostics
bash scripts/run_two_model_experiment.sh
bash scripts/run_mixed_concurrent_experiment.sh
```

Results are stored in `result.proposed` inside the generated JSON report.
Diagnostic runs also include observability snapshots and timing breakdowns.
