#!/usr/bin/env python3
"""Train and freeze the P4 router without opening calibration labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = rank(left), rank(right)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return -1.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recovery-config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P4 trainer refuses to run outside {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise SystemExit("P4 router training requires CUDA to be hidden")
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from torch.nn import functional as functional

    from savr.pair.features import feature_schema_sha256
    from savr.pair.router import PairRouter
    from savr.pair.router_torch import TorchPairRouter, frozen_adamw, pinball_loss
    from savr.pair.types import HORIZONS, ONSET_LAYERS

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    recovery = json.loads((ROOT / args.recovery_config).read_text(encoding="utf-8"))
    run_root = args.run_root.resolve()
    if not run_root.is_relative_to(ROOT / "results"):
        raise SystemExit("P4 trainer run root is outside project results")
    immutable_outputs = (
        run_root / "router_checkpoint.json",
        run_root / "router_training_summary.json",
        run_root / "router_artifact.json",
    )
    if any(path.exists() for path in immutable_outputs):
        raise SystemExit("P4 immutable router output already exists")
    worker_path = run_root / "worker_summary.json"
    if not worker_path.is_file():
        raise SystemExit("P4 completed worker summary is required")
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    if worker.get("status") != "completed" or worker.get("run_id") != recovery["run_id"]:
        raise SystemExit("P4 worker summary is incomplete or mismatched")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    # These are the only label-bearing paths opened by the trainer.
    train_intervention_path = run_root / "train_interventions.jsonl"
    train_feature_path = run_root / "train_features.jsonl"
    train_contract_path = run_root / "train_contracts.jsonl"
    if (
        file_sha256(train_intervention_path) != worker["interventions_sha256"]["train"]
        or file_sha256(train_feature_path) != worker["features_sha256"]["train"]
        or file_sha256(train_contract_path) != worker["contracts_sha256"]["train"]
    ):
        raise SystemExit("P4 training artifacts changed")
    interventions = {row["record_id"]: row for row in read_jsonl(train_intervention_path)}
    features = read_jsonl(train_feature_path)
    contracts = {
        row["contract_id"]: row
        for row in read_jsonl(train_contract_path)
        if row["branch_kind"] == "base" and row["category"] != "all_fresh_control"
    }
    by_contract: dict[str, list[dict[str, Any]]] = {}
    for row in features:
        by_contract.setdefault(row["contract_id"], []).append(row)
    if set(by_contract) != set(contracts) or len(features) != 640 or len(contracts) != 256:
        raise SystemExit("P4 training population accounting changed")
    for rows in by_contract.values():
        rows.sort(key=lambda row: row["query_ordinal"])

    def internal_fit(contract: Mapping[str, Any]) -> bool:
        value = hashlib.sha256(
            f"{config['router']['internal_split_seed']}|{contract['trajectory_id']}".encode()
        ).digest()
        return int.from_bytes(value[:8], "big") / 2**64 < float(
            config["router"]["internal_train_fit_fraction"]
        )

    fit_ids = sorted(key for key, value in contracts.items() if internal_fit(value))
    validation_ids = sorted(set(contracts) - set(fit_ids))
    if len(fit_ids) < 180 or len(validation_ids) < 35:
        raise SystemExit("P4 internal trajectory split is too small")

    fit_feature_rows = [row for key in fit_ids for row in by_contract[key]]
    group_values = np.asarray(
        [group["continuous"] for row in fit_feature_rows for group in row["features"]["groups"]],
        dtype=np.float64,
    )
    global_values = np.asarray(
        [row["features"]["global_continuous"] for row in fit_feature_rows], dtype=np.float64
    )
    group_mean, group_scale = group_values.mean(0), group_values.std(0)
    global_mean, global_scale = global_values.mean(0), global_values.std(0)
    group_scale[group_scale < 1e-8] = 1.0
    global_scale[global_scale < 1e-8] = 1.0
    fit_contracts = [contracts[key] for key in fit_ids]
    signed_mean = float(np.mean([row["signed_contract_regret"] for row in fit_contracts]))
    signed_scale = max(
        float(np.std([row["signed_contract_regret"] for row in fit_contracts])), 1e-8
    )
    positive_scale = max(
        float(np.std([row["positive_contract_regret"] for row in fit_contracts])), 1e-8
    )
    distortion_scale = max(
        float(np.std([row["mean_action_l1_distortion"] for row in fit_contracts])), 1e-8
    )
    target_stats = {
        "signed_mean": signed_mean,
        "signed_scale": signed_scale,
        "positive_scale": positive_scale,
        "distortion_scale": distortion_scale,
    }

    def tensors(row: Mapping[str, Any]) -> tuple[Any, ...]:
        groups = row["features"]["groups"]
        continuous = torch.tensor(
            (np.asarray([item["continuous"] for item in groups]) - group_mean) / group_scale,
            dtype=torch.float32,
        )
        return (
            continuous,
            torch.tensor([0 if item["camera"] == "primary" else 1 for item in groups], dtype=torch.long),
            torch.tensor([item["tile"] for item in groups], dtype=torch.long),
            torch.tensor([ONSET_LAYERS.index(item["onset_layer"]) for item in groups], dtype=torch.long),
            torch.zeros(len(groups), dtype=torch.long),
            torch.tensor([HORIZONS.index(item["horizon"]) for item in groups], dtype=torch.long),
            torch.tensor(
                (np.asarray(row["features"]["global_continuous"]) - global_mean) / global_scale,
                dtype=torch.float32,
            ),
        )

    def contract_loss(model: Any, contract_id: str) -> Any:
        contract = contracts[contract_id]
        outputs = []
        group_losses = []
        for row in by_contract[contract_id]:
            output = model(*tensors(row))
            outputs.append(output)
            if contract["category"] == "atomic":
                target = interventions[row["base_intervention_record_id"]]["signed_regret"]
                group_losses.append(
                    functional.huber_loss(
                        output.signed_group_regret,
                        torch.full_like(
                            output.signed_group_regret,
                            (target - signed_mean) / signed_scale,
                        ),
                        delta=1.0,
                    )
                )
        signed_prediction = torch.stack([item.signed_contract_regret for item in outputs]).mean()
        q50_prediction = torch.stack([item.positive_regret_q50 for item in outputs]).mean()
        q90_prediction = torch.stack([item.positive_regret_q90 for item in outputs]).mean()
        distortion_prediction = torch.stack(
            [item.dense_action_distortion for item in outputs]
        ).mean()
        signed_target = torch.tensor(
            (contract["signed_contract_regret"] - signed_mean) / signed_scale,
            dtype=torch.float32,
        )
        positive_target = torch.tensor(
            contract["positive_contract_regret"] / positive_scale, dtype=torch.float32
        )
        distortion_target = torch.tensor(
            contract["mean_action_l1_distortion"] / distortion_scale, dtype=torch.float32
        )
        loss = functional.huber_loss(signed_prediction, signed_target, delta=1.0)
        if group_losses:
            loss = loss + 0.5 * torch.stack(group_losses).mean()
        loss = loss + 0.25 * pinball_loss(q50_prediction, positive_target, 0.5)
        loss = loss + 0.5 * pinball_loss(q90_prediction, positive_target, 0.9)
        loss = loss + 0.25 * functional.huber_loss(
            distortion_prediction, distortion_target, delta=1.0
        )
        return loss

    def evaluate(model: Any, ids: Sequence[str]) -> float:
        model.eval()
        with torch.no_grad():
            return float(np.mean([contract_loss(model, key).item() for key in ids]))

    trained = {}
    seed_summaries = {}
    for seed in config["router"]["training_seeds"]:
        torch.manual_seed(int(seed))
        model = TorchPairRouter((config["profile"]["profile_id"],), seed=int(seed))
        optimizer = frozen_adamw(model)
        best_loss = float("inf")
        best_state = None
        patience = 0
        for epoch in range(int(config["router"]["maximum_epochs"])):
            model.train()
            ordered = sorted(
                fit_ids,
                key=lambda key: hashlib.sha256(f"{seed}|{epoch}|{key}".encode()).hexdigest(),
            )
            for start in range(0, len(ordered), int(config["router"]["batch_size_contracts"])):
                batch = ordered[start : start + int(config["router"]["batch_size_contracts"])]
                optimizer.zero_grad()
                loss = torch.stack([contract_loss(model, key) for key in batch]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["router"]["gradient_clip_norm"])
                )
                optimizer.step()
            validation_loss = evaluate(model, validation_ids)
            if validation_loss < best_loss - float(config["router"]["minimum_delta"]):
                best_loss = validation_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= int(config["router"]["patience"]):
                break
        if best_state is None:
            raise RuntimeError("P4 router training produced no finite checkpoint")
        model.load_state_dict(best_state)
        trained[int(seed)] = model
        seed_summaries[str(seed)] = {
            "validation_composite": best_loss,
            "epochs": epoch + 1,
        }
    best_value = min(value["validation_composite"] for value in seed_summaries.values())
    eligible = [
        int(seed)
        for seed, value in seed_summaries.items()
        if value["validation_composite"] <= best_value * 1.01
    ]
    selected_seed = min(eligible)
    selected = trained[selected_seed]

    def contract_proxy(contract_id: str, name: str) -> float:
        rows = by_contract[contract_id]
        if name == "seeded_random":
            return int(hashlib.sha256(f"proxy|{contract_id}".encode()).hexdigest()[:16], 16) / 16**16
        index = {
            "source_age": 0,
            "raw_change": 1,
            "projected_change": 6,
            "proprio_change": 12,
            "previous_action_change": 14,
            "retained_salience": 16,
        }[name]
        return float(
            np.mean(
                [
                    np.mean([group["continuous"][index] for group in row["features"]["groups"]])
                    for row in rows
                ]
            )
        )

    proxy_candidates = []
    observed = [contracts[key]["signed_contract_regret"] for key in validation_ids]
    for name in config["router"]["proxy_candidates"]:
        values = [contract_proxy(key, name) for key in validation_ids]
        forward = spearman(values, observed)
        reverse = spearman([-value for value in values], observed)
        proxy_candidates.append(
            {"name": name, "orientation": 1 if forward >= reverse else -1, "spearman": max(forward, reverse)}
        )
    selected_proxy = max(proxy_candidates, key=lambda row: (row["spearman"], row["name"]))

    state = selected.state_dict()
    weights = {
        "camera_embedding": state["camera_embedding.weight"].numpy(),
        "tile_embedding": state["tile_embedding.weight"].numpy(),
        "layer_embedding": state["layer_embedding.weight"].numpy(),
        "profile_embedding": state["profile_embedding.weight"].numpy(),
        "horizon_embedding": state["horizon_embedding.weight"].numpy(),
        "group_w1": state["group_encoder.0.weight"].numpy().T,
        "group_b1": state["group_encoder.0.bias"].numpy(),
        "group_w2": state["group_encoder.2.weight"].numpy().T,
        "group_b2": state["group_encoder.2.bias"].numpy(),
        "group_head_w": state["group_head.weight"].numpy().T,
        "group_head_b": state["group_head.bias"].numpy(),
        "set_w1": state["set_encoder.0.weight"].numpy().T,
        "set_b1": state["set_encoder.0.bias"].numpy(),
        "set_w2": state["set_encoder.2.weight"].numpy().T,
        "set_b2": state["set_encoder.2.bias"].numpy(),
        "set_head_w": state["set_head.weight"].numpy().T,
        "set_head_b": state["set_head.bias"].numpy(),
    }
    router = PairRouter(
        profiles=(config["profile"]["profile_id"],),
        seed=selected_seed,
        group_mean=group_mean,
        group_scale=group_scale,
        global_mean=global_mean,
        global_scale=global_scale,
        weights=weights,
    )
    checkpoint_path = run_root / "router_checkpoint.json"
    checkpoint_sha256 = router.save(checkpoint_path)
    training_summary = {
        "schema_version": "pair-p4-router-training-v1",
        "run_id": recovery["run_id"],
        "selected_seed": selected_seed,
        "seed_summaries": seed_summaries,
        "fit_contracts": len(fit_ids),
        "validation_contracts": len(validation_ids),
        "target_statistics": target_stats,
        "group_mean": group_mean.tolist(),
        "group_scale": group_scale.tolist(),
        "global_mean": global_mean.tolist(),
        "global_scale": global_scale.tolist(),
        "proxy_candidates": proxy_candidates,
        "selected_proxy": selected_proxy,
        "feature_schema_sha256": feature_schema_sha256(),
        "training_label_paths_opened": [
            train_intervention_path.name,
            train_contract_path.name,
        ],
        "calibration_label_paths_opened": [],
        "checkpoint_sha256": checkpoint_sha256,
    }
    training_summary["semantic_sha256"] = hashlib.sha256(
        canonical_bytes(training_summary)
    ).hexdigest()
    training_summary_path = run_root / "router_training_summary.json"
    write_once(training_summary_path, training_summary)
    sealed_hash = hashlib.sha256(
        canonical_bytes(
            {
                "interventions": worker["interventions_sha256"]["calibration"],
                "features": worker["features_sha256"]["calibration"],
                "contracts": worker["contracts_sha256"]["calibration"],
            }
        )
    ).hexdigest()
    artifact = {
        "schema_version": "pair-router-artifact-v1",
        "freeze_sha256": recovery["semantic_sha256"],
        "project_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dataset_revision": "f13aa24a3da8c43c7225569f28c562979fa0e35a",
        "split_manifest_sha256": "2f0131608bacd74d71df82de07f852fcaec315783f67ba550a663da56b899d56",
        "feature_schema_sha256": feature_schema_sha256(),
        "architecture": "group-64-32/deepsets-64-32/gelu/no-dropout",
        "training_seed": selected_seed,
        "training_statistics_sha256": file_sha256(training_summary_path),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": sealed_hash,
        "created_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    }
    artifact["artifact_id"] = hashlib.sha256(canonical_bytes(artifact)).hexdigest()
    write_once(run_root / "router_artifact.json", artifact)
    print(json.dumps({"status": "frozen", "selected_seed": selected_seed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
