# Lab 3 — Containerizing the model with Docker

Files: `src/food11/serve.py`, `Dockerfile`, `.dockerignore`. The image measurements below were taken on a GitHub Actions `ubuntu-latest` runner on 2026-11-17.

## Register your best model

The best Lab 2 run was registered as **`food11`** and given the alias **`champion`**:

```python
mlflow.register_model("runs:/<run-id>/model", "food11")
MlflowClient().set_registered_model_alias("food11", "champion", 1)
```

Champion run `2a5c6111…`: mini dataset, lr = 1e-4, batch size 32, 5 epochs, `val_accuracy` = 0.7299, `test_accuracy` = 0.7536.

## Question 1 — Version number; logged model vs registered model

It got **version 1** (the first version of a new registered model).

- **A run's logged model** is just an artifact of *one* run (`mlflow.pytorch.log_model(model, "model")`): weights, `MLmodel` file, environment. It is identified by the run (or, in MLflow 3, a `models:/m-<id>` logged-model id) and has no lifecycle.
- **A registered model** is a named entry in the Model Registry (`food11`) that collects *versions*. Each version points to one logged model and can carry aliases, tags and a description. Consumers ask for `models:/food11@champion` and never need to know which run produced it.

## Question 2 — Aliases instead of stages

The built-in stages `None / Staging / Production / Archived` are deprecated. They were replaced by **aliases**: free-named, mutable pointers such as **`champion`** and **`challenger`** (Lab 5's `promote.py` uses both), plus **tags** for any other metadata (e.g. `gate_decision`, `validation_status`).

**Why version separately from the run.** A run is an experiment record: params, metrics, artifacts. Many runs never matter. The registry gives the few that do a stable name, a version history, an audit trail of who promoted what, and a place for review and approval, all without tying serving code to run IDs.

**Why an alias is more flexible than a stage.**
- You can have as many aliases as you need, with your own names (`champion`, `challenger`, `shadow`, `eu-prod`…), instead of four fixed ones.
- Each alias points to exactly one version, so `@champion` is unambiguous. With stages, several versions could be "Production" at once.
- Moving an alias is an atomic, instant switch that the serving code picks up on its next load, and rolling back is just moving it again.

## Question 3 — Why load through a model URI?

`models:/food11@champion` separates the serving code from *which* model is served:

- **No path to a file.** The code doesn't depend on a `.pth` path that exists only on my laptop. MLflow downloads the artifacts from the tracking server, so the same code works in a container or on another machine.
- **The right loader comes with the model.** `pyfunc` loads it with the flavour, preprocessing contract and dependencies recorded at logging time; there is no `torch.load` + rebuilding the architecture by hand.
- **Lineage.** The served model is traceable to its run, params and metrics.

**To serve a newer version: change nothing in the code.** Move the alias (`set_registered_model_alias("food11", "champion", <new version>)`) and restart the service. With a `.pth` path I would have to copy the new file, edit or rebuild, and redeploy.

## Question 4 — Why copy `pyproject.toml`/`uv.lock` and `uv sync` before the source?

Docker caches each layer and reuses it as long as the instruction *and the files it copies* haven't changed. Once one layer changes, every layer after it is rebuilt.

```dockerfile
COPY pyproject.toml uv.lock ./                          # changes rarely
RUN uv sync --frozen --no-dev --no-install-project      # slow: 1.4 GB of wheels
...
COPY src ./src                                          # changes often
```

**Measured** after appending one line to `serve.py` and rebuilding:

```
#6  [builder 2/5] WORKDIR /app                                   CACHED
#9  [builder 3/5] RUN pip install uv                             CACHED
#7  [builder 4/5] COPY pyproject.toml uv.lock ./                 CACHED
#8  [builder 5/5] RUN uv sync --frozen --no-dev ...              CACHED
#10 [stage-1 3/4] COPY --from=builder /app/.venv /app/.venv      CACHED
#11 [stage-1 4/4] COPY src ./src                                 ← only this ran
rebuild took 5s
```

With a single `COPY . .` before `uv sync`, any edit to any file invalidates the dependency layer, and every code change would mean re-downloading and installing torch, mlflow etc. (minutes instead of seconds).

## Question 5 — Single-stage vs multi-stage image size

| Image | Size |
|---|---|
| naive single-stage (`FROM python:3.12`, `COPY . .`, `pip install uv && uv sync` in the same image) | **3.58 GB** |
| repo `Dockerfile` (multi-stage, `python:3.12-slim` runtime) | **2.00 GB** (−44 %) |

`docker history food11-api:multi` (largest layers):

```
1.44GB   COPY /app/.venv /app/.venv          ← torch (CPU), torchvision, mlflow, pandas, scipy...
87.7MB   debian base (slim)
41.4MB   python 3.12 install
57.3kB   COPY src ./src
```

`docker history food11-api:single` (largest layers):

```
1.55GB   RUN pip install uv && uv sync ...   ← venv + uv itself + pip cache
696MB    apt-get: gcc, build tools, headers   ← full python image, not needed at runtime
202MB    apt-get: ...dev libraries
134MB    debian base (full)
```

The dependencies themselves (≈1.4 GB, mostly torch) are the same in both. Multi-stage saves about 1.6 GB by **not shipping the build environment**: compilers and `-dev` packages of the full Python image, uv, and pip caches. This is also why CPU-only torch matters: the CUDA build would add several GB to the 1.44 GB layer.

## Question 6 — Forgetting `.dockerignore`

Without it, `docker build` sends the **whole folder** to the Docker daemon as the build context before building anything:

| Folder | Size here |
|---|---|
| `data/` | 1.27 GB (36,578 files) |
| `mlartifacts/` | ~130 MB of models |
| `.venv/` | Windows virtual environment (torch alone is several hundred MB) |
| `.git/`, `mlflow.db`, `__pycache__` | also sent |

- **Build speed:** every build first uploads well over 1.5 GB ("Sending build context…"), even if only one line of code changed.
- **Image size:** anything reached by `COPY` is baked into the image. With this Dockerfile only `COPY src` copies from the context, so the size impact is small, but a `COPY . .` would put the dataset and the models in the image.

**Which folders would break the build?**
- **`.venv/`**, if it were copied (e.g. `COPY . .` after `uv sync`, or into `/app`). It would overwrite the Linux virtual environment built in the image with my **Windows** `.venv` (`Scripts/`, `.exe`, `.pyd` files), and the container would fail with missing `python`/`uvicorn`.
- **`__pycache__`**, compiled for another Python version, is harmless.
- **`data/`** and **`mlartifacts/`** don't break anything; they make the build very slow and can leak data into the image.

## Question 7 — Why not `127.0.0.1:5000` from inside the container?

A container has its **own network namespace**. Inside it, `127.0.0.1` is the *container's* loopback, and nothing is listening on port 5000 there, so the connection is refused. The MLflow server runs on the *host's* loopback.

`host.docker.internal` is a special DNS name that Docker Desktop (Mac/Windows) resolves to the **host machine's IP as seen from the container** (the gateway of Docker's internal VM network). On Linux, `--network host` instead makes the container share the host's network namespace, so `127.0.0.1` *is* the host.

## Question 8 — New container from the same image: does the model still load?

**Yes.** A new container from the same image loads the model again without a rebuild, as long as the MLflow server is reachable. If `champion` was moved meanwhile, the new container even loads the *new* version.

So the image contains the **code and the dependencies** (Python, torch, mlflow, FastAPI, `serve.py`), but **not the model**. The model is **fetched at startup** from the registry (`models:/food11@champion` → download artifacts → `pyfunc.load_model`). That is the design: image and model are versioned and released independently. The cost is that the container can't start without a reachable registry; see Lab 4 Q6.

## Question 9 — What's missing to run the exact image elsewhere?

The image exists only in my local Docker. Another machine needs:

1. **A registry** to pull from (Docker Hub, GHCR, ECR…), plus credentials if it is private.
2. **An immutable tag** tied to the source, e.g. the **git commit SHA**, rather than `latest`, so "the exact image" means something.
3. **An automated, reproducible build**: CI building from a clean checkout on every push, not from my laptop's state.

These are now in place (Lab 5): GitHub Actions builds and tests the image on every push and pushes `ghcr.io/hovigtav/mlops-lab-1:<commit-sha>` on `main`. Branch protection ensures only tested code reaches `main`.
