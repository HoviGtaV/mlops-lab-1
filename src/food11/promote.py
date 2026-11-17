"""
src/food11/promote.py

Quality gate for the model registry. Compares a candidate model version
with the current `champion` and only moves the alias when the candidate is
clearly better (not just better by noise):

    promote  if  acc_candidate >= acc_champion + 2 * se
    se = sqrt(p * (1 - p) / n),  p = acc_champion,  n = size of the test split

Otherwise the candidate gets the `challenger` alias. Either way, the decision
and the numbers are written as tags on the candidate version.

    uv run python src/food11/promote.py --version 2
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

DATASET_DIRS = {
    "mini": Path("data/food11_processed_mini"),
    "processed": Path("data/food11_processed"),
}


def standard_error(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def decide(acc_candidate: float, acc_champion: float, n: int) -> tuple[bool, float]:
    """Returns (promote?, threshold the candidate had to reach)."""
    threshold = acc_champion + 2 * standard_error(acc_champion, n)
    return acc_candidate >= threshold, threshold


def test_split_size(dataset: str) -> int:
    folder = DATASET_DIRS[dataset] / "evaluation"
    n = sum(1 for p in folder.rglob("*") if p.is_file())
    if n == 0:
        raise SystemExit(f"No images found in {folder}; pass --n explicitly")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a model version if it beats the champion")
    parser.add_argument("--version", required=True, help="candidate model version number")
    parser.add_argument("--model-name", default="food11")
    parser.add_argument("--metric", default="test_accuracy")
    parser.add_argument("--n", type=int, help="test split size (default: counted from data/)")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    client = MlflowClient(tracking_uri=tracking_uri)
    name, metric = args.model_name, args.metric

    candidate = client.get_model_version(name, args.version)
    cand_run = client.get_run(candidate.run_id)
    acc_cand = cand_run.data.metrics[metric]

    try:
        champion = client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        champion = None

    if champion is None:
        client.set_registered_model_alias(name, "champion", candidate.version)
        client.set_model_version_tag(
            name, candidate.version, "gate_decision", "promoted (no champion)"
        )
        client.set_model_version_tag(
            name, candidate.version, "gate_candidate_acc", f"{acc_cand:.4f}"
        )
        print(
            f"No champion yet: version {candidate.version} is now champion ({metric}={acc_cand:.4f})"
        )
        return

    if champion.version == candidate.version:
        raise SystemExit(f"Version {candidate.version} is already the champion")

    champ_run = client.get_run(champion.run_id)
    acc_champ = champ_run.data.metrics[metric]

    cand_data = cand_run.data.params.get("dataset")
    champ_data = champ_run.data.params.get("dataset")
    if cand_data != champ_data:
        print(
            f"WARNING: candidate was tested on '{cand_data}', champion on '{champ_data}'. "
            "The accuracies come from different test sets."
        )

    n = args.n or test_split_size(champ_data or "mini")
    se = standard_error(acc_champ, n)
    promote, threshold = decide(acc_cand, acc_champ, n)

    if promote:
        client.set_registered_model_alias(name, "champion", candidate.version)
        decision = f"promoted (replaced v{champion.version})"
    else:
        client.set_registered_model_alias(name, "challenger", candidate.version)
        decision = f"rejected (champion v{champion.version} kept)"

    tags = {
        "gate_decision": decision,
        "gate_candidate_acc": f"{acc_cand:.4f}",
        "gate_champion_acc": f"{acc_champ:.4f}",
        "gate_champion_version": str(champion.version),
        "gate_se": f"{se:.4f}",
        "gate_threshold": f"{threshold:.4f}",
        "gate_n": str(n),
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, candidate.version, key, value)

    print(f"candidate v{candidate.version}: {metric} = {acc_cand:.4f}")
    print(f"champion  v{champion.version}: {metric} = {acc_champ:.4f}")
    print(f"n = {n}, se = {se:.4f}, threshold = acc_champion + 2*se = {threshold:.4f}")
    print(f"decision: {decision}")


if __name__ == "__main__":
    main()
