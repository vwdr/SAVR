"""Read-only CPU diagnosis of the dense execution boundary; no compressed analysis."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path("/home/ved/SAVR")


def verified(path, expected):
    raw = (ROOT / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"evidence identity differs: {path}")
    return raw.decode()


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError("project root required")
    config = json.loads(verified("configs/openvla/original_baseline_40task_v1.json",
        "90a65540dfade976ad58924b8318576b38c310ac1d8831db31ceb3d4682daa27"))
    path = config["checkpoint"] + "/modeling_prismatic.py"
    source = verified(path, config["authenticated_files"][path])
    # The released loader overrides config.json's pretraining norm_stats with
    # this fine-tuning statistics file. Use the actual LIBERO runtime values.
    cfg_path = config["checkpoint"] + "/dataset_statistics.json"
    metadata = json.loads(verified(cfg_path, config["authenticated_files"][cfg_path]))
    method = next(n for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.FunctionDef) and n.name == "_unnormalize_actions")
    # Execute only this pure NumPy method, not the checkpoint module or loader.
    namespace = dict(np=np, ACTION_PROPRIO_NORMALIZATION_TYPE="bounds_q99",
                     NormalizationType=SimpleNamespace(BOUNDS="bounds", BOUNDS_Q99="bounds_q99"))
    exec(compile(ast.Module(body=[method], type_ignores=[]), path, "exec"), namespace)
    synthetic = np.linspace(-.9, .9, 56, dtype=np.float32).reshape(8, 7)
    examples = []
    for key, value in sorted(metadata.items()):
        if not key.startswith("libero_"):
            continue
        owner = SimpleNamespace(get_action_stats=lambda _key, stats=value["action"]: stats)
        raw = namespace["_unnormalize_actions"](owner, synthetic.copy(), key)
        baseline = np.asarray(raw, dtype=np.float32)
        bridge = np.asarray(raw)
        if raw.dtype != np.float64 or bridge.dtype != np.float64 or baseline.dtype != np.float32:
            raise ValueError("expected dtype discrepancy not reproduced")
        examples.append(dict(statistics_key=key, normalized_dtype=str(synthetic.dtype),
            checkpoint_output_dtype=str(raw.dtype), baseline_execution_dtype=str(baseline.dtype),
            new_bridge_execution_dtype=str(bridge.dtype),
            max_abs_rounding_difference=float(np.max(np.abs(bridge-baseline.astype(np.float64))))))
    records = json.loads(verified("results/specprune-episode-qualification-v01/records.json",
        "fe47fbd39c0263138dc6ce87f9aad98b50fde54367ccdc5740bbbac99b044963"))
    dense = [r for r in records["episodes"] if r["mode"] == "dense"]
    if len(dense) != 1:
        raise ValueError("unique dense record required")
    print(json.dumps(dict(dense_record=dense[0],
        offline_dense_identity_max_abs=max(v for row in records["offline"]
                                           for errors in row["dense_parity"] for v in errors.values()),
        synthetic_only_not_recorded_actions=examples, statistics_file=cfg_path,
        compressed_outcome_inspected=False,
        gpu_used=False, confirmed_cause_of_step_difference=False), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
