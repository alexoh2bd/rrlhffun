# Run SWE-bench harness: before/after RL comparison on matched 50-instance subset.
#
# Usage:
#   python harness.py baseline   → evaluates predictions_base_eval50.json  (pre-RL)
#   python harness.py rl         → evaluates predictions_rl_eval50.json     (post-RL)
#   python harness.py both       → runs baseline then rl sequentially
#
# Both evals use the same 50 instance IDs so results are directly comparable.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _ensure_docker_host() -> None:
    """Docker Desktop on macOS uses ~/.docker/run/docker.sock."""
    if os.environ.get("DOCKER_HOST"):
        return
    sock = Path.home() / ".docker/run/docker.sock"
    if sock.is_socket():
        os.environ["DOCKER_HOST"] = f"unix://{sock}"


def _verify_docker_reachable() -> None:
    """Fail fast with a clear message if Docker is not usable."""
    _ensure_docker_host()
    try:
        r = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        sys.exit("Docker CLI not found in PATH. Install Docker Desktop.")
    except subprocess.TimeoutExpired:
        sys.exit("Docker did not respond. Is Docker Desktop running?")
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip() or "unknown error"
        sys.exit(f"Cannot reach Docker daemon.\nDetails: {msg}")


_verify_docker_reachable()

from swebench.harness.run_evaluation import main as run_eval  # noqa: E402


DATASET_NAME = "princeton-nlp/SWE-bench_oracle"
SPLIT        = "test"

# Pre-built matched prediction files (50 instances, valid patches in both models)
CONFIGS = {
    "baseline": {
        "predictions_path": "predictions_baseline_100.json",
        "run_id":           "baseline_eval_v1",
    },
    "rl": {
        "predictions_path": "predictionsRL_100.json",
        "run_id":           "rl_eval_v1",  # reuses the 50 already completed
    },
}


def run_one(mode: str) -> None:
    cfg = CONFIGS[mode]
    path = Path(cfg["predictions_path"])
    if not path.exists():
        sys.exit(f"Predictions file not found: {path}. Run the setup script first.")

    with path.open() as f:
        preds = json.load(f)
    print(f"\n{'='*60}")
    print(f"Running {mode.upper()} eval: {len(preds)} instances")
    print(f"  predictions : {path}")
    print(f"  run_id      : {cfg['run_id']}")
    print(f"  model       : {preds[0].get('model_name_or_path','?')}")
    print(f"{'='*60}\n")

    run_eval(
        dataset_name     = DATASET_NAME,
        split            = SPLIT,
        predictions_path = str(path),
        max_workers      = 4,
        force_rebuild    = False,
        cache_level      = "env",
        clean            = False,
        open_file_limit  = 4096,
        run_id           = cfg["run_id"],
        timeout          = 1_800,
        namespace        = "swebench",
        rewrite_reports  = False,
        modal            = False,
        instance_ids     = None,
    )


def print_comparison() -> None:
    """Print a side-by-side summary from the two report JSON files."""
    import glob
    results = {}
    for mode, cfg in CONFIGS.items():
        pattern = f"*{cfg['run_id']}*.json"
        files = glob.glob(pattern)
        if not files:
            results[mode] = None
            continue
        with open(sorted(files)[-1]) as f:
            results[mode] = json.load(f)

    print("\n" + "="*60)
    print("BEFORE vs AFTER RL — SWE-bench Oracle (50 matched instances)")
    print("="*60)
    header = f"{'Metric':<35} {'Baseline':>10} {'RL-tuned':>10}"
    print(header)
    print("-"*60)

    keys = [
        ("submitted_instances",   "Submitted"),
        ("completed_instances",   "Completed (ran)"),
        ("resolved_instances",    "Resolved ✓"),
        ("unresolved_instances",  "Unresolved ✗"),
        ("error_instances",       "Errors (patch fail)"),
        ("empty_patch_instances", "Empty patches"),
    ]
    for key, label in keys:
        bv = results["baseline"].get(key, "N/A") if results["baseline"] else "N/A"
        rv = results["rl"].get(key, "N/A") if results["rl"] else "N/A"
        print(f"  {label:<33} {str(bv):>10} {str(rv):>10}")

    # Resolve rates
    for mode in ("baseline", "rl"):
        r = results[mode]
        if r and r.get("submitted_instances", 0):
            rate = r["resolved_instances"] / r["submitted_instances"] * 100
            print(f"\n  Resolve rate ({mode}): {rate:.1f}%")
    print("="*60)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode not in ("baseline", "rl", "both", "compare"):
        sys.exit("Usage: python harness.py [baseline|rl|both|compare]")

    if mode == "compare":
        print_comparison()
    elif mode == "both":
        run_one("baseline")
        run_one("rl")
        print_comparison()
    else:
        run_one(mode)
