#!/usr/bin/env python3
"""Analyze only an authenticated completed contemporaneous-reference bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from savr.openvla.contemporary_contract import DESIGN_SHA, SUITES, reconcile, require


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and len(a) > 0 and np.isfinite(a).all(), "invalid descriptive sample")
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)))


def analyze(design, reference, summary, records, observation_ids, *, resamples=10000):
    reconcile(design, reference, summary, records, observation_ids)
    require(type(resamples) is int and resamples >= 100, "too few bootstrap draws")
    pairs, controls, suite_results = [], [], []
    for suite in SUITES:
        rows = [r for r in records["episodes"] if r["suite"] == suite]
        native = [r for r in rows if r["arm"] == "native"]
        ids = list(dict.fromkeys(r["condition_id"] for r in rows if r["arm"] != "native"))
        local_pairs = []
        for identity in ids:
            ds = {r["arm"]: r for r in rows if r["condition_id"] == identity and r["arm"] != "native"}
            entry = dict(condition_id=identity, suite=suite, dense=ds["dense"], specprune=ds["specprune"],
                         success_difference=int(ds["specprune"]["success"]) - int(ds["dense"]["success"]))
            pairs.append(entry)
            local_pairs.append(entry)
        sentinel = local_pairs[0]["dense"]["success"]
        repeatability_limited = len({sentinel, *(r["success"] for r in native)}) != 1
        native_traces = [[(p["observation_sha256"], p["command_sha256"])
                          for p in records["parity"] if p["slot_id"] == r["slot_id"]] for r in native]
        controls.append(dict(suite=suite, before=native[0], after=native[1],
                             observation_command_traces_equal=native_traces[0] == native_traces[1],
                             repeatability_limited=repeatability_limited))
        suite_results.append(dict(suite=suite, conditions=10, repeatability_limited=repeatability_limited,
                                  dense_successes=sum(p["dense"]["success"] for p in local_pairs),
                                  specprune_successes=sum(p["specprune"]["success"] for p in local_pairs)))
    rng = np.random.default_rng(7)
    differences = np.array([[p["success_difference"] for p in pairs if p["suite"] == s] for s in SUITES])
    sampled = np.zeros(resamples)
    for suite in differences:
        sampled += suite[rng.integers(0, 10, (resamples, 10))].sum(axis=1) / 40
    timing = [r for r in records["timing"] if not r["warmup"]]
    timing_stats = {arm: {label: stats([r["seconds"] for r in timing if r["arm"] == arm
                                       and (frame is None or r["frame"] == frame)])
                         for label, frame in (("first_query", 0), ("second_query", 1), ("combined", None))}
                    for arm in ("dense", "specprune")}
    # Cluster all rounds/frames by trajectory; never bootstrap 128 independent tasks.
    means = {arm: np.array([np.mean([r["seconds"] for r in timing
                                     if r["arm"] == arm and r["observation_id"] == identity])
                            for identity in observation_ids]) for arm in ("dense", "specprune")}
    selections = rng.integers(0, 8, (resamples, 8))
    savings = 1 - means["specprune"][selections].mean(axis=1) / means["dense"][selections].mean(axis=1)
    episode_costs = {arm: {"episode_seconds": stats([p[arm]["episode_seconds"] for p in pairs]),
                           "policy_queries": stats([p[arm]["policy_queries"] for p in pairs]),
                           "query_seconds": stats([q["seconds"] for p in pairs for q in p[arm]["queries"]]),
                           "executed_steps": stats([p[arm]["executed_steps"] for p in pairs]),
                           "total_replans": sum(p[arm]["replans"] for p in pairs),
                           "total_discarded_actions": sum(p[arm]["discarded_actions"] for p in pairs)}
                     for arm in ("dense", "specprune")}
    return dict(schema_version="contemporary-reference-analysis-v1", technical_reconciliation_passed=True,
                positive_method_result=False, population="40 consumed development conditions; seed 7",
                dense_successes=sum(p["dense"]["success"] for p in pairs),
                specprune_successes=sum(p["specprune"]["success"] for p in pairs),
                historical_dense_successes=39, historical_dense_conditions=40,
                specprune_only_successes=sum(p["success_difference"] == 1 for p in pairs),
                dense_only_successes=sum(p["success_difference"] == -1 for p in pairs),
                success_difference=float(differences.mean()),
                descriptive_paired_success_interval_95=np.quantile(sampled, [.025, .975]).tolist(),
                bootstrap=dict(seed=7, draws=resamples, success_unit="task, stratified by suite",
                               timing_unit="trajectory, all paired rounds/frames resampled together"),
                pairs=pairs, native_controls=controls, suites=suite_results, episode_costs=episode_costs,
                timing_seconds=timing_stats,
                controlled_trace_mean_time_reduction=float(1 - means["specprune"].mean() / means["dense"].mean()),
                descriptive_trace_reduction_interval_95=np.quantile(savings, [.025, .975]).tolist(),
                limitations=["No proposed corrector was trained or tested.",
                             "Controlled coarse traces are not a deployment speedup estimate.",
                             "Development intervals are not significance or noninferiority claims.",
                             "Two native repeats per suite are diagnostic, not precise variance estimates.",
                             "Any weaker dense baseline must be investigated before improvement claims."])


def load_bundle(root, run):
    root, run = root.resolve(), run.resolve()
    require(run.is_relative_to(root / "results"), "only project result bundles may be analyzed")
    require((run / "worker_summary.json").is_file() and not (run / "technical_stop.json").exists(),
            "completed immutable summary required; stopped runs are not analyzable")
    summary = json.loads((run / "worker_summary.json").read_text())
    require(summary.get("complete") is True, "worker summary is not completed")
    for relative, expected in summary["artifacts_sha256"].items():
        path = (run / relative).resolve()
        require(path.is_relative_to(run) and sha(path) == expected, "artifact hash mismatch")
    require({"launch.json", "records.json", "progress.jsonl", "loaded_runtime.json"}
            <= set(summary["artifacts_sha256"]), "required evidence not authenticated")
    launch = json.loads((run / "launch.json").read_text())
    config = launch["config"]
    require(config.get("schema_version") == "contemporary-reference-executable-v1"
            and config.get("mode") == "evaluation" and config.get("launch_ready") is True,
            "not a frozen primary evaluation")
    for relative, expected in config["authenticated_files"].items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and sha(path) == expected, "frozen source/config changed")
    design_path = root / "configs/openvla/contemporary_reference_design_v1.json"
    require(sha(design_path) == DESIGN_SHA, "design identity changed")
    design = json.loads(design_path.read_text())
    refpath = root / design["reference_config"]
    require(sha(refpath) == design["reference_config_sha256"], "reference identity changed")
    require(config["analyzer_sha256"] == sha(Path(__file__)), "analyzer identity changed")
    manifest = json.loads((root / design["timing"]["observation_manifest"]).read_text())
    return design, json.loads(refpath.read_text()), summary, json.loads((run / "records.json").read_text()), [r["trajectory_id"] for r in manifest["inputs"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = analyze(*load_bundle(root, args.run))
    output = args.run / "analysis.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("technical_reconciliation_passed", "dense_successes", "specprune_successes", "positive_method_result")}))


if __name__ == "__main__":
    main()
