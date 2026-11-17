# Lab 2 — Model training and experiment tracking with MLflow

Files: `src/food11/train.py`, `pyproject.toml`, `uv.lock`, `.gitignore`.

> **Note on the runs below.** The original Lab 2 comparison runs were lost when MLflow moved into Docker Compose in Lab 4: the registry was started fresh, and only the run that became `champion` was kept. The four comparison runs were **re-run on 2026-11-17** with the same script, settings and seed (42), on CPU, against the current registry, where they are visible in the `food11` experiment. The champion run `2a5c6111` (lr 1e-4, bs 32, 5 epochs) is the original Lab 2 best run.

## Question 1 — What changed in `pyproject.toml` and `uv.lock`?

`uv add mlflow torch torchvision scikit-learn` (commit `38d4dbf`):

- **`pyproject.toml`:** the four packages were added to `[project] dependencies` with *lower bounds* of the versions resolved at that time:
  ```toml
  dependencies = [
      "mlflow>=3.16.1",
      "pillow>=12.3.0",
      "scikit-learn>=1.9.1",
      "torch>=2.14.0",
      "torchvision>=0.29.0",
  ]
  ```
- **`uv.lock`:** grew by about 2,900 lines. It pins the **exact** version, download URL and SHA-256 hash of *every* package in the dependency tree (~100: mlflow's Flask/SQLAlchemy/pandas/pyarrow…, torch's sympy/networkx…, numpy, scipy…), with platform markers. `pyproject.toml` says what the project *needs*; `uv.lock` records exactly *what was installed*, so `uv sync --frozen` reproduces it anywhere.

In Lab 3 the `[[tool.uv.index]]` *pytorch-cpu* source was added, so torch/torchvision resolve to the CPU wheels (`2.14.0+cpu`) instead of the multi-GB CUDA build.

## Question 2 — `--backend-store-uri` vs `--default-artifact-root`

- **`--backend-store-uri sqlite:///mlflow.db`**: where the **metadata** goes, i.e. experiments, runs, params, metrics (every value with its step and timestamp), tags, and the model registry (names, versions, aliases). Small, structured, queryable (searching runs, sorting by `val_accuracy`, comparing). Stored in a database: SQLite here, PostgreSQL/MySQL in production.
- **`--default-artifact-root ./mlruns`**: where the **artifacts** go, i.e. *files* such as the trained model (`model.pth`, `MLmodel`, `requirements.txt`, `conda.yaml`), plots and data samples. Large binary blobs, stored on a filesystem or object store (S3, GCS, Azure Blob).

The metadata *describes* a run and points to its artifacts (each run has an `artifact_uri`); the artifacts are the run's actual outputs.

From Lab 4 on, the server uses `--artifacts-destination … --serve-artifacts` instead: clients upload and download artifacts *through* the server (`mlflow-artifacts:/` URIs), so they never need direct filesystem access to it.

## Question 3 — Why keep `mlflow.db` and `mlruns/` out of git *and* DVC?

- **Not git:** they are **outputs**, not source code. They change on every run; the database is a binary file that can't be diffed or merged; artifacts are hundreds of MB of models (git history keeps every version forever); and two people training in parallel would get conflicts on the same SQLite file.
- **Not DVC:** DVC versions *inputs* and pipeline outputs that you want to **check out together with a commit** (the dataset). MLflow already *is* the versioning system for runs and models: every run has an ID, every registered model has versions and aliases. Snapshotting `mlflow.db` with DVC would make `dvc checkout` roll the whole experiment history back to an older state, and the right way to share runs between people is a **shared tracking server** (or a hosted one like DagsHub's), not copying the database around.

## Question 4 — `set_experiment` with a new name

MLflow **creates the experiment** and makes it active: `food11` appeared in the UI next to "Default", with **experiment ID 1** ("Default" is 0). Calling it again with the same name just reuses the existing experiment, so the script can be run any number of times.

## Question 5 — `log_param` vs `log_metric`

- **`log_param`** records an **input** fixed for the whole run: `dataset`, `epochs`, `lr`, `batch_size`, `model`, `num_classes`, `seed`. It is written once; logging the same key again with a different value is an error.
- **`log_metric`** records an **output** that can be measured many times: `train_loss`, `val_loss`, `val_accuracy` per epoch, then `test_accuracy` and `test_loss` at the end.

`log_metric` takes a **`step`** because a metric is a **time series**: one value per epoch (`step=epoch`). This is what lets the UI draw learning curves and spot over-fitting (val loss going up while train loss keeps falling). A param has a single value, so there is nothing to order.

## Question 6 — Params, charts, and where the model lives

In the run page:
- **Overview:** the 7 params and the final metric values.
- **Model metrics:** a chart for each of `train_loss`, `val_loss`, `val_accuracy` over 5 steps.
- **Artifacts / Logged models:** the `model` logged with `mlflow.pytorch.log_model`.

On disk the model is a folder of files under the artifact store, not in the database. With the original Lab 2 server it was `./mlruns/1/<run_id>/artifacts/model/`. With the current server (MLflow 3 logged models, `--artifacts-destination ./mlartifacts`):

```
mlartifacts/1/models/m-bb170b6952fe477491b8c652f4820f50/artifacts/
├── MLmodel                    # flavour (pytorch + pyfunc), signature, how to load it
├── conda.yaml / python_env.yaml / requirements.txt   # environment
└── data/
    ├── model.pth              # the pickled model (≈43 MB)
    └── pickle_module_info.txt
```

The database only stores the run's `artifact_uri` (`mlflow-artifacts:/1/…`) pointing there.

## Question 7 — Which learning rate gave the best `val_accuracy`?

Mini dataset, ResNet-18 pretrained on ImageNet, Adam, 5 epochs, seed 42:

| lr | batch | val_accuracy (final) | val_accuracy per epoch | test_accuracy |
|---|---|---|---|---|
| 0.01 | 32 | **0.163** | 0.108 → 0.119 → 0.149 → 0.156 → 0.163 | 0.172 |
| 0.001 | 32 | **0.479** | 0.223 → 0.386 → 0.573 → 0.519 → 0.479 | 0.509 |
| **0.0001** | 32 | **0.726** | 0.651 → 0.711 → 0.714 → 0.722 → 0.726 | **0.756** |
| 0.001 | 64 | **0.574** | 0.406 → 0.477 → 0.579 → 0.569 → 0.574 | 0.590 |
| *(Lab 5)* 0.00001 | 32 | 0.507 | — | 0.525 |

**lr = 0.0001 is clearly best.** **Higher is not better.** At **0.01** the updates are so large that they destroy the pretrained ImageNet features (val loss 32 after epoch 1). The model is barely above chance (1/11 ≈ 0.09) and never recovers. At **0.001** it learns, but it is unstable: val accuracy peaks at epoch 3 (0.573), then falls while train loss keeps dropping. Too *low* is also bad: **1e-5** (trained in Lab 5) only reaches 0.51 in 5 epochs, because it learns too slowly. Accuracy against learning rate is an **inverted U**.

## Question 8 — Parallel coordinates (`lr`, `batch_size`, `val_accuracy`)

- The lines from **lr = 1e-4** all end at the top of the `val_accuracy` axis, the ones from **1e-2** at the bottom, and **1e-3** in between. Learning rate is the parameter that matters most here.
- **Batch size has a smaller, interacting effect.** At lr = 1e-3, going from 32 to 64 raises val accuracy from 0.48 to 0.57. A bigger batch gives a less noisy gradient and half as many updates per epoch, which partly calms an lr that is too high. It cannot fix a bad lr.
- With one run per setting and a single seed, small differences are noisy. Re-running the champion's exact settings gave 0.726 instead of 0.730 val accuracy (non-deterministic CPU kernels). This is the reason for the `2·se` margin in the Lab 5 promotion gate.

## Question 9 — Best run

Sorted by `val_accuracy` descending, the best Lab 2 run is **`2a5c61113eca4fc39758780dbc0b57f0`** (lr 1e-4, batch 32, 5 epochs; val_accuracy 0.7299, test_accuracy 0.7536). Its re-run (`9a9de094…`, val_accuracy 0.7263) comes just below. That run ID was registered in Lab 3 as **`food11` version 1** and given the alias **`champion`**.

(Lab 5 later added v3: the same lr for 10 epochs, val 0.7418 / test 0.7646. The quality gate did *not* promote it, because the +1.1-point gain is within noise; see `lab/lab5.md`.)
