# mlops-lab-1 — Food-11 classifier, from data to CI/CD

A ResNet-18 that classifies food photos into 11 categories, built up over five MLOps labs:

| Lab | Topic | Main files | Write-up |
|---|---|---|---|
| 1 | git + DVC data versioning (DagsHub remote) | `data.dvc`, `.dvc/config`, `src/food11/data.py` | [lab/lab1.md](lab/lab1.md) |
| 2 | Training + MLflow experiment tracking | `src/food11/train.py` | [lab/lab2.md](lab/lab2.md) |
| 3 | Model registry + Docker serving API | `src/food11/serve.py`, `Dockerfile` | [lab/lab3.md](lab/lab3.md) |
| 4 | Docker Compose: MLflow + API + frontend | `docker-compose.yml`, `mlflow/`, `frontend/` | [lab/lab4.md](lab/lab4.md) |
| 5 | CI/CD with GitHub Actions, data contract, promotion gate | `.github/workflows/ci.yml`, `tests/`, `src/food11/validate_data.py`, `src/food11/promote.py` | [lab/lab5.md](lab/lab5.md) |

## Setup

```bash
uv sync                 # Python 3.12, CPU-only torch
dvc pull                # data/ from the DagsHub remote (needs credentials in .dvc/config.local)
```

## Common commands

```bash
# data
uv run python src/food11/data.py                                   # raw -> processed + mini (128x128)
uv run python src/food11/validate_data.py data/food11_processed_mini

# MLflow + serving stack (MLflow UI :5000, frontend :8501)
docker compose up -d --build

# train (logs to MLflow) and optionally register a new food11 version
uv run python src/food11/train.py --dataset mini --epochs 5 --lr 0.0001 --batch-size 32 --register

# quality gate: move `champion` only if clearly better, else `challenger`
uv run python src/food11/promote.py --version <n>

# serve the new champion
docker compose restart inference

# checks run by CI
uv run ruff check src tests
uv run pytest -q
```

## Pipeline

Every push and pull request runs `test` (ruff + pytest) then `smoke` (build the image, start it with `SKIP_MODEL=1`, probe `/health`). On `main`, the image is pushed as `ghcr.io/hovigtav/mlops-lab-1:<commit-sha>`. `main` is protected: changes go through a pull request, and `test` and `smoke` must pass.

The code version is the image's commit SHA. The model version is the registry alias `models:/food11@champion`, read when the API starts.
