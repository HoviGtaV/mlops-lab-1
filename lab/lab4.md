# Lab 4 — Orchestrating the app with Docker Compose

Files: `mlflow/Dockerfile`, `frontend/Dockerfile`, `frontend/app.py`, `docker-compose.yml`. The behaviour quoted below (container logs, `docker compose ps`, persistence tests) was recorded on a GitHub Actions `ubuntu-latest` runner on 2026-11-17, using this repo's own images and compose file.

## Differences from the lab sheet (on purpose)

| Lab sheet | This repo | Why |
|---|---|---|
| named volume `mlflow-data:/mlflow-data` | **bind mounts** `./mlflow.db:/mlflow-data/mlflow.db` and `./mlartifacts:/mlflow-data/mlartifacts` | keeps using the registry from Labs 2–3 (runs, `food11` versions, `champion` alias) and keeps it visible and backed up in the project folder (both paths are git-ignored) |
| `--default-artifact-root /mlflow-data/mlruns` | `--artifacts-destination /mlflow-data/mlartifacts --serve-artifacts` | the server *proxies* artifact uploads and downloads (`mlflow-artifacts:/…` URIs). Clients on the host and in other containers never need direct access to the server's filesystem, which a container path like `/mlflow-data/mlruns` would require |
| `pip install mlflow` | `mlflow==3.16.1` | same version as the client in `uv.lock`; an unpinned server could upgrade the DB schema on a rebuild |
| `--allowed-hosts` not mentioned | `localhost`, `127.0.0.1`, `mlflow` (port 5000) | MLflow 3 rejects requests with an unexpected `Host` header (DNS-rebinding protection), so `http://mlflow:5000` from the inference container must be allowed explicitly |
| — | `restart: on-failure` on `inference` | see Question 6 |
| `Staging` stage | `champion` alias | stages are deprecated (Lab 3); `serve.py` loads `models:/food11@champion` |

## Question 1 — Running the mlflow image without a volume

Everything written to `/mlflow-data` goes to the container's **writable layer**, which belongs to that one container and is deleted with it (`docker rm`). A new container from the same image starts from the image's clean filesystem.

Measured:

```
container 1: create experiment "persist-test"
  experiments -> ['persist-test', 'Default']
docker stop m1 && docker rm m1
container 2 (same image):
  experiments -> ['Default']          <- the experiment is gone
```

The UI of the second container shows only the empty "Default" experiment: no runs, no registered models.

## Question 2 — Why a named volume rather than a bind mount?

- **Named volume:** storage created and managed by Docker (under Docker's own data directory), independent of the project folder and the host's path layout.
  - Portable: the compose file works the same on Windows, Mac and Linux, with no host path to get wrong.
  - Docker sets the right ownership on first use.
  - Docker Desktop on Windows/Mac stores it *inside* its Linux VM, which is much faster for SQLite than a mount of a Windows folder.
  - It can't be deleted or edited by accident from the file explorer or by `git clean`.
  - `docker compose down -v` gives a deliberate way to reset it.
- **Bind mount:** maps a host folder or file. Convenient for local development (you can see and back up the files, and reuse an existing `mlflow.db`). But it depends on the host path and permissions, and it has one classic trap: **if `./mlflow.db` doesn't exist yet, Docker creates a *directory* with that name, and MLflow fails to start.**

**Would a bind mount work just as well?** Functionally yes, and this repo uses bind mounts so the Lab 2–3 registry carries over. But they are not equivalent: the bind mount is less portable, slower on Docker Desktop, needs the files to exist first, and is *not* removed by `down -v` (see Question 10).

## Question 3 — Why does `http://mlflow:5000` resolve now?

`docker compose up` creates a **user-defined bridge network** for the project (`mlops-lab-1_default`) and attaches all three services to it. Docker runs an **embedded DNS server** (127.0.0.11) on such networks that resolves each **service name** to that container's IP. So `mlflow` resolves to the mlflow container, and port 5000 is reached directly on the private network.

In Lab 3 the API container ran alone on the default bridge network, where there is no name-based DNS, and MLflow wasn't in a container at all: it ran on the host. Hence `host.docker.internal` (Docker Desktop's name for the host) or `--network host`.

## Question 4 — Why is `INFERENCE_URL` an environment variable?

The service address is a property of **where the container runs**, not of the code. Compose sets `INFERENCE_URL=http://inference:8000`. But a `docker run` of the frontend image on its own is not on the Compose network, so the name `inference` doesn't resolve there.

Reading it from the environment (default `http://127.0.0.1:8000`) means **the same image** works in Compose, standalone (`-e INFERENCE_URL=http://host.docker.internal:8000`), in Kubernetes, or in staging vs production, with no rebuild. That is the "config in the environment" rule of twelve-factor apps. `MLFLOW_TRACKING_URI` in `serve.py` works the same way.

## Question 5 — Why doesn't `inference` publish a port?

Only **humans** need to reach a service from the host: the MLflow UI (5000) and the Streamlit page (8501). The inference API is only called by the frontend, and that traffic stays **inside the Compose network**: the frontend's server-side code calls `http://inference:8000`, the container's own port, through Docker DNS. A published port is only needed to enter the network from the host.

Not publishing it also means a smaller attack surface (the unauthenticated `/predict` isn't exposed on the machine's interfaces) and no host port conflicts.

## Question 6 — What happens if MLflow isn't ready when `inference` starts?

`depends_on` only orders **container start**: the inference container starts as soon as the mlflow *process* has been launched. `serve.py` loads `models:/food11@champion` at startup. If MLflow isn't accepting connections yet, `mlflow.pyfunc.load_model` raises (connection refused or retries exhausted). The FastAPI startup fails, uvicorn exits with an error, and **the container stops**. It stays stopped with `Exited (3)` and nothing restarts it, so the frontend gets connection errors on every upload. `docker compose logs inference` shows the traceback.

The same happens if the registry is reachable but has no champion. Measured with an empty registry:

```
inference-1  | INFO:     Started server process [1]
inference-1  | mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: Registered Model with name=food11 not found
```

**Fix applied:** `restart: on-failure` on `inference`. Docker restarts the container with an increasing back-off until MLflow is ready and the model loads. In the test above it restarted **6 times in 45 s** (`RestartCount 6`), and it recovers by itself as soon as the model exists. A stricter alternative is a `healthcheck` on mlflow (`curl -f localhost:5000/health`) with `depends_on: mlflow: condition: service_healthy`, plus a retry loop in `serve.py`.

## Question 7 — `docker compose ps`

```
SERVICE     STATUS          PORTS
frontend    Up 45 seconds   0.0.0.0:8501->8501/tcp, [::]:8501->8501/tcp
inference   Up 2 seconds    8000/tcp
mlflow      Up 46 seconds   0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp
```

**Published:** `mlflow` (5000) and `frontend` (8501), shown as `0.0.0.0:host->container`.

**Not published:** `inference` shows only `8000/tcp`. That is the port exposed *inside* the network (from `EXPOSE 8000`), with no host mapping.

This matches `docker-compose.yml`: only the two services with a `ports:` section are reachable from the host. ("Up 2 seconds" for inference is the restart loop from Question 6, since that test registry was empty.)

## Question 8 — After promoting a new version: which model answers?

**The old one.** `serve.py` loads the model **once, at startup**, into the module-level `model`. Moving the `champion` alias in the registry doesn't change the object already in memory, so refreshing the frontend still gets predictions from the previous version.

The single command to pick up the new champion without rebuilding anything:

```bash
docker compose restart inference
```

## Question 9 — Why does `restart` alone work?

On restart the container runs `serve.py` again, which resolves `models:/food11@champion` **at that moment** and downloads whatever version the alias now points to.

So the **image** contains only the code and the dependencies (Python, torch, mlflow, FastAPI, `serve.py`). **The model is not baked in**: it is fetched from the registry at container startup. A new model never needs a new image, and a new image never changes the model. The two are released independently (Lab 5: image tag = commit SHA, model = `champion` alias).

## Question 10 — Persistence across `down` / `up` and `down -v`

Measured with this compose file:

```
create experiment "survives?"
docker compose down && docker compose up -d     -> ['survives?', 'Default']
docker compose down -v && docker compose up -d  -> ['survives?', 'Default']
```

- **`down` / `up`:** yes, the registered model and the `champion` alias are still there. `down` removes the containers and the network, but the data lives outside the containers.
- **`down -v`:** with the lab sheet's **named volume**, `-v` deletes `mlflow-data`. The next `up` starts with an empty database: no runs, no `food11`, no alias, and inference fails as in Question 6.
  **In this repo nothing is lost**, because the data is in **bind mounts** (`./mlflow.db`, `./mlartifacts`). `-v` only removes named and anonymous volumes declared by the project, never host folders. To reset this registry you would delete the files yourself.

## Question 11 — What Compose can't give you

Compose runs containers on **one Docker host** and does not manage them across machines.

- **Three `inference` replicas behind a load balancer.** `docker compose up --scale inference=3` starts three containers on the same machine, with Docker DNS round-robin. But Compose has no real load balancer with health-based routing, no rolling updates (zero-downtime replacement of replicas), no autoscaling on load, and no spreading across machines.
- **MLflow surviving a machine failure.** It needs:
  - several nodes and a scheduler that **reschedules** the container on a healthy node,
  - storage that is **not tied to one disk**: a network or managed database (PostgreSQL/MySQL instead of SQLite) and object storage (S3/GCS/MinIO) for artifacts,
  - health checks with self-healing, and a stable service address that follows the container.

That is what an orchestrator such as **Kubernetes** provides: Deployments with replicas, Services and Ingress for load balancing, liveness and readiness probes, rolling updates, HorizontalPodAutoscaler, and PersistentVolumes or managed services for state.
