# Lab 1 — git/dvc and data preparation

Repository: <https://github.com/HoviGtaV/mlops-lab-1> · DVC remote: DagsHub `jean.madjarian/mlops-lab-1`

**DVC remote solution adopted:** none of the two fallbacks. The full dataset (raw + processed + mini, 36,578 files, 1.27 GB) is pushed to the **DagsHub** DVC remote. The HTTPS remote (`https://dagshub.com/<user>/mlops-lab-1.dvc`) was first replaced by DagsHub's **S3-compatible endpoint**, which handles a large push better (commit `7d1feaa`):

```ini
['remote "origin"']
    url = s3://dvc
    endpointurl = https://dagshub.com/jean.madjarian/mlops-lab-1.s3
```

## Question 1 — Files created by `uv init`

| File | Content |
|---|---|
| `pyproject.toml` | Project metadata: name, version, authors, `requires-python = ">=3.12"`, the `dependencies` list (empty at first), the build backend (`uv_build`) and a console script `mlops-lab-1 = "mlops_lab_1:main"` |
| `.python-version` | `3.12`, the interpreter version uv uses for the project's virtual environment |
| `README.md` | Empty project readme |
| `src/mlops_lab_1/__init__.py` | Package skeleton with a `main()` that prints "Hello from mlops-lab-1!" |

Created on the first `uv add` / `uv run`:
- **`uv.lock`:** the exact resolved version, source URL and hash of every direct and transitive dependency, so every machine installs the same packages.
- **`.venv/`:** the local virtual environment. It is not committed.

## Question 2 — Files created by `dvc init`

| File | Purpose | Push to git? |
|---|---|---|
| `.dvc/config` | Project-level DVC config: default remote, remote URL | **Yes** (no secrets in it) |
| `.dvc/.gitignore` | Tells git to ignore DVC's private files: `/config.local`, `/tmp`, `/cache` | **Yes** |
| `.dvcignore` | Like `.gitignore`, but for files DVC should not track | **Yes** |
| `.dvc/cache/` | Content-addressed local copy of every tracked file version (created at `dvc add`) | No |
| `.dvc/tmp/` | DVC's internal state and lock files | No |
| `.dvc/config.local` | Machine-specific config, e.g. credentials | **Never** |

Commit `f03ef7c` pushed exactly `.dvc/.gitignore`, `.dvc/config` and `.dvcignore`.

## Question 3 — Where are the credentials stored?

DVC has four config levels:

| Flag | File | Scope |
|---|---|---|
| `--system` | e.g. `C:\ProgramData\iterative\dvc\config` | every user on the machine |
| `--global` | `%LOCALAPPDATA%\iterative\dvc\config` (Linux: `~/.config/dvc/config`) | every project of this user |
| *(default)* `--project` | `.dvc/config` | this repo, **committed to git** |
| `--local` | `.dvc/config.local` | this repo on this machine, **git-ignored** |

With `--global`, as in the lab, the credentials go in the user-level file outside the repo. After switching to the S3 endpoint, my DagsHub token (`access_key_id` / `secret_access_key`) is in `.dvc/config.local`, which `.dvc/.gitignore` excludes.

**The credentials must never be pushed to GitHub.** Anyone who can read the repo could then read *and overwrite* the dataset. I checked the full git history: no password, key or token was ever committed.

## Question 4 — What happened to `.gitignore`?

`dvc add data` added the line **`/data`** to `.gitignore`. DVC now owns the folder: it hashes the files, stores them in `.dvc/cache` and writes a small pointer file. Git is told to ignore the real data, so 1.2 GB of images never end up in git history.

## Question 5 — The `.dvc` file

Yes, **`data.dvc`**. It is the *pointer* that git versions in place of the data. After the first `dvc add` (raw data only, commit `7e05652`):

```yaml
outs:
- md5: a3a457d03c51ff8b037a833440f6ad13.dir   # hash of the folder's file list
  size: 1188442712                              # 1.19 GB
  nfiles: 16643
  hash: md5
  path: data
```

After adding the processed datasets (commit `cf47c3d`): `md5: 94afad23…dir`, `size: 1274882585`, `nfiles: 36578`. The `.dir` hash points to a manifest in the cache/remote that lists every file with its own MD5.

## Question 6 — What is on GitHub, and what is on DagsHub?

- **GitHub** has the code and the configuration (`pyproject.toml`, `.dvc/config`, `.gitignore`), plus **`data.dvc`**, the file that points to the data. **It does not have the images.**
- **DagsHub** shows the data: `dvc push` uploaded the content-addressed files that `data.dvc` refers to.

Git versions the *pointer*; DVC versions the *data*.

## Question 7 — Cloning into a new folder

After `git clone`, **there is no `data/` folder**. It is git-ignored, and only `data.dvc` comes with the clone. To get it:

```bash
dvc pull        # = dvc fetch (download from the remote into the cache) + dvc checkout
```

## Question 8 — Checking out an older commit

```bash
git log --oneline -- data.dvc
#   cf47c3d Add food11_processed and food11_processed_mini
#   7e05652 Track data folder with dvc
git checkout 7e05652
dvc checkout
```

**No, the new folders are gone.** After `dvc checkout` on `7e05652`, `data/` contains only `food11_raw/`, because that commit's `data.dvc` points to the 16,643-file raw-only version. `git checkout main` + `dvc checkout` brings back `food11_processed/` and `food11_processed_mini/`. The code and the data move together, each to the version recorded in the same commit.

## Data preparation (`src/food11/data.py`)

The script reads `data/food11_raw/<split>/<class_id>_<n>.jpg`, takes the class from the file name (`0` → Bread … `10` → Vegetable-Fruit), resizes every image to **128×128 RGB** and writes them in the `ImageFolder` layout that the torchvision ResNet pipeline expects:

| Dataset | training | validation | evaluation |
|---|---|---|---|
| `food11_raw` | 9,866 | 3,430 | 3,347 |
| `food11_processed` | 9,866 | 3,430 | 3,347 |
| `food11_processed_mini` (≤100 per class) | 1,100 | 1,096 | 1,096 |

(Rice has only 96 images in validation and evaluation of the mini set, which is why those splits are short of 1,100.)

Note found in Lab 5: the original Kaggle data contains **byte-identical images in more than one split**. The mini dataset has 43 such images, for example `training/0_106.jpg` = `evaluation/0_131.jpg`. See `lab/lab5.md`, Question 3.
