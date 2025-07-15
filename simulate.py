# simulation.py
"""
Self‑contained runner that integrates:
- Hydra for configuration & auto run directories
- Git auto‑commit if working tree is dirty
- Unique run_hash = <git_hash>-<uuid8>
- Result saved as HDF5 (h5py)
- Environment snapshot (conda env or pip freeze)
- Optional lightweight ZIP of tracked source files (.py/.yaml/.ipynb) when dirty
- MLflow logging (params, metrics, tags, artifacts)

Folder layout expected:
project/
 ├ simulation.py  <-- this file
 ├ configs/
 │   └ config.yaml
 └ mlruns/        <-- created automatically (central store)

Example config.yaml
-------------------
alpha: 0.5
run_id: ${uuid:8}

(hydra section can stay default; Hydra will create outputs/<date>/<time>/)
"""

from __future__ import annotations

# --- Standard library
import datetime as dt
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import uuid
from zipfile import ZipFile, ZIP_DEFLATED

# --- Third‑party
import git                                     # pip install gitpython
import h5py                                    # pip install h5py
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import mlflow                                  # pip install mlflow
import numpy as np

# ---------------------------------------------------------------------------
# Helper: auto‑commit dirty tree (commit message stamped by timestamp)
# ---------------------------------------------------------------------------

def ensure_clean_commit(repo: git.Repo) -> tuple[str, bool]:
    """Commit all staged + unstaged + untracked changes if repo is dirty.

    Returns
    -------
    git_hash : str
        8‑char hexsha of HEAD after the operation.
    is_dirty : bool
        True  if a commit was created (dirty state existed).
    """
    if repo.is_dirty(untracked_files=True):
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        msg = f"run(auto): {ts}"
        repo.git.add(all=True)               # stage everything (tracked/untracked)
        repo.index.commit(msg)
        return repo.head.commit.hexsha[:8], True
    return repo.head.commit.hexsha[:8], False

# ---------------------------------------------------------------------------
# Helper: snapshot conda / pip environment to a file and return path
# ---------------------------------------------------------------------------

def snapshot_env(out_dir: pathlib.Path) -> pathlib.Path:
    """Export current Python env (conda or pip) into a file inside out_dir."""
    env_file = out_dir / ("environment.yml" if os.getenv("CONDA_DEFAULT_ENV") else "requirements.txt")
    try:
        if os.getenv("CONDA_DEFAULT_ENV"):
            subprocess.run([
                "conda", "env", "export", "--no-builds", "-n", os.getenv("CONDA_DEFAULT_ENV")
            ], check=True, stdout=env_file.open("w"))
        else:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, stdout=env_file.open("w"))
    except Exception as e:
        # Fallback minimal info
        env_file.write_text(f"# export failed: {e}\n")
    return env_file

# ---------------------------------------------------------------------------
# Helper: create a ZIP of tracked source files with certain extensions
# ---------------------------------------------------------------------------

def zip_tracked_sources(repo: git.Repo, out_dir: pathlib.Path, extensions=(".py", ".yaml", ".ipynb")) -> pathlib.Path:
    zip_path = out_dir / f"src_{uuid.uuid4().hex[:6]}.zip"
    tracked_files = [p for p in repo.git.ls_files().splitlines() if pathlib.Path(p).suffix in extensions]
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for rel in tracked_files:
            zf.write(rel, arcname=rel)
    return zip_path

# ---------------------------------------------------------------------------
# Main hydra entry
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Locate repo (search_parent_directories works even after Hydra chdir)
    repo = git.Repo(search_parent_directories=True)

    # 2. Commit if dirty & grab git_hash
    git_hash, made_commit = ensure_clean_commit(repo)
    dirty_flag = "dirty" if made_commit else "clean"

    # 3. Unique uuid for this run
    uuid8 = uuid.uuid4().hex[:8]
    run_hash = f"{git_hash}-{uuid8}"

    # 4. Hydra output directory
    out_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5. Run simple simulation using alpha
    alpha = float(cfg.alpha)
    data = np.linspace(0, 1, 11) * alpha
    h5_path = out_dir / "result.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("array", data=data)
        f.attrs.update({"alpha": alpha, "git_hash": git_hash, "uuid": uuid8})

    # 6. Snapshot environment
    env_file = snapshot_env(out_dir)

    # 7. If original state was dirty, zip sources (optional; keeps repo lean)
    zip_path: pathlib.Path | None = None
    if dirty_flag == "dirty":
        zip_path = zip_tracked_sources(repo, out_dir)

    # 8. MLflow logging
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("sim_demo")

    with mlflow.start_run(run_name=run_hash):
        mlflow.set_tag("git_hash", git_hash)
        mlflow.set_tag("uuid8", uuid8)
        mlflow.set_tag("dirty", dirty_flag)
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("mean", float(data.mean()))
        mlflow.log_artifact(h5_path)
        mlflow.log_artifact(env_file)
        if zip_path is not None:
            mlflow.log_artifact(zip_path, artifact_path="source_zip")

    print(f"✔ Run complete. run_hash={run_hash}\n   outputs: {out_dir}\n   MLflow run saved.")


if __name__ == "__main__":
    main()

