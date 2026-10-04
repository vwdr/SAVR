"""Offline coverage vs adequacy diagnostic for current-frame visual correction.

Read-only CPU analysis of ALREADY-SAVED evidence. No GPU, no new data, no robot
rollouts. Does not modify any frozen source, summary, weight or evidence file.

Question: does visual-correction failure (worse L1 than base on 260/800 validation
samples, and on 77/144 samples where base L1 < 0.02) correspond better to
(1) observations far from the fitting distribution in visual/state space
   ("coverage" hypothesis), or
(2) alteration of already-adequate base actions regardless of coverage
   ("always-on correction" hypothesis)?

Metrics (all predeclared here):
- Scene vector per record: mean of the 512 current visual patch embeddings
  (4096-dim, BF16 decoded to FP32) concatenated with the 8-dim normalized state.
- Instruction vector: the already-pooled 4096-dim instruction embedding.
- Each dimension z-scored using ONLY the 3,200 fitting records; the SAME fit-only
  mean/std are applied to both matrices. (v01 erroneously standardized fit and
  validation with their own per-matrix statistics, making distances inconsistent;
  that output is superseded.)
- OOD score per validation record: Euclidean distance to its nearest fitting
  neighbor in scene space (k1), mean of 10 nearest (k10), and the same two for
  instruction space.
- Adequacy: base (compression-only) L1 vs dense teacher on the same observation.
- Residual magnitude: mean absolute difference |visual_prediction - base_prediction|
  over the 8x7 command grid.

Predeclared interpretation (descriptive only, no significance claim):
- Let Q1/Q4 = validation records in the lowest/highest scene-OOD quartile.
  Coverage dominates if the visual improvement rate (visual_l1 < base_l1) in Q4 is
  >=10 percentage points below Q1, and this holds after stratifying by base
  adequacy. Adequacy dominates if OOD-quartile improvement rates move little while
  records with small base L1 show systematically lower (or negative) improvement
  regardless of OOD.

Hard limits: this validation pool is demonstration-derived held-out trajectories,
NOT states reached by the corrected policy; result bounds do not measure the
deployment distribution shift (that requires new per-step rollout data).
"""
import json
import os
import sys
import zipfile
import numpy as np

ROOT = "/Users/veddwivedi/Documents/VLA/SAVR"
COLLECTION = os.path.join(ROOT, "results/current-frame-feature-collection-v01")
FIT = os.path.join(ROOT, "results/current-frame-adapter-fit-v01")
OUT = os.path.join(ROOT, "results/current-frame-correction-coverage-diagnostic-v02")

BF16_FIELDS = {"action_features", "current_visual", "instruction"}


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def decode_record(path):
    """Return (meta, scene_vec(4104,), instruction(4096,)) streaming, no full copy kept."""
    with np.load(path, allow_pickle=False) as npz:
        meta = json.loads(npz["metadata"].tobytes())
        canned = {}
        for name in ("current_visual", "instruction", "state"):
            arr = npz[name]
            if name in BF16_FIELDS:
                arr = bf16_to_f32(arr)
            else:
                arr = arr.astype(np.float32)
            if name == "current_visual":
                canned["visual_mean"] = arr.mean(axis=0)
            elif name == "instruction":
                canned["instruction"] = arr
            else:
                canned["state"] = arr
    scene = np.concatenate([canned["visual_mean"], canned["state"]])
    return meta, scene, canned["instruction"]


def main():
    os.makedirs(OUT, exist_ok=True)
    records = json.load(open(os.path.join(COLLECTION, "records.json")))
    by_sample = {}
    for r in records:
        if r["sample_id"] in by_sample:
            raise SystemExit("duplicate sample_id in records.json")
        by_sample[r["sample_id"]] = (r["artifact"], r["learning_role"])

    validation = json.load(open(os.path.join(FIT, "validation.json")))
    if len(validation) != 800:
        raise SystemExit(f"expected 800 validation rows, got {len(validation)}")
    for v in validation:
        if v["sample_id"] not in by_sample:
            raise SystemExit(f"validation sample {v['sample_id']} missing from collection")

    fit_ids = [sid for sid, (_, role) in by_sample.items() if role == "fit"]
    val_ids = [sid for sid, (_, role) in by_sample.items() if role == "validation"]
    if len(fit_ids) != 3200 or len(val_ids) != 800:
        raise SystemExit(f"role counts unexpected: fit={len(fit_ids)} val={len(val_ids)}")

    # Streaming pass: build scene + instruction matrices.
    fit_scene = np.empty((len(fit_ids), 4104), dtype=np.float32)
    fit_inst = np.empty((len(fit_ids), 4096), dtype=np.float32)
    val_scene = np.empty((len(val_ids), 4104), dtype=np.float32)
    val_inst = np.empty((len(val_ids), 4096), dtype=np.float32)
    val_meta = {}

    fit_index = {sid: i for i, sid in enumerate(fit_ids)}
    val_index = {sid: i for i, sid in enumerate(val_ids)}

    for i, (sid, (artifact, role)) in enumerate(by_sample.items()):
        path = os.path.join(COLLECTION, artifact)
        meta, scene, inst = decode_record(path)
        if meta["sample_id"] != sid or meta["split"] != "train":
            raise SystemExit(f"metadata mismatch at {artifact}")
        idx = fit_index.get(sid)
        if idx is not None:
            fit_scene[idx] = scene
            fit_inst[idx] = inst
        else:
            idx = val_index[sid]
            val_scene[idx] = scene
            val_inst[idx] = inst
        if i % 400 == 0:
            print(f"...decoded {i}/4000", flush=True)
    print("decoded 4000/4000")

    # z-score using FIT-ONLY statistics, applied identically to fit and validation.
    def zscore_with(mat, mean, std):
        std = np.where(std < 1e-6, 1.0, std)
        return (mat - mean) / std

    fit_mean_scene = fit_scene.mean(axis=0)
    fit_std_scene = fit_scene.std(axis=0)
    fit_scene_z = zscore_with(fit_scene, fit_mean_scene, fit_std_scene)
    val_scene_z = zscore_with(val_scene, fit_mean_scene, fit_std_scene)
    fit_mean_inst = fit_inst.mean(axis=0)
    fit_std_inst = fit_inst.std(axis=0)
    fit_inst_z = zscore_with(fit_inst, fit_mean_inst, fit_std_inst)
    val_inst_z = zscore_with(val_inst, fit_mean_inst, fit_std_inst)

    # Precomputed assisted kNN squared distances.
    def knn_distances(query, ref):
        q2 = (query ** 2).sum(1)
        r2 = (ref ** 2).sum(1)
        cross = query @ ref.T
        d2 = q2[:, None] + r2[None, :] - 2.0 * cross
        np.maximum(d2, 0.0, out=d2)
        return d2

    d_scene = knn_distances(val_scene_z, fit_scene_z)   # (800, 3200)
    d_inst = knn_distances(val_inst_z, fit_inst_z)
    k1_scene = d_scene.min(1)
    k10_scene = np.partition(d_scene, 9, axis=1)[:, :10].mean(1)
    k1_inst = d_inst.min(1)
    k10_inst = np.partition(d_inst, 9, axis=1)[:, :10].mean(1)

    # Assemble per-sample rows joined with fit metrics.
    rows = []
    for i, v in enumerate(validation):
        sid = v["sample_id"]
        delta_visual_base = v["visual_l1"] - v["base_l1"]
        delta_ao_base = v["action_only_l1"] - v["base_l1"]
        delta_shuf_base = v["visual_shuffled_l1"] - v["base_l1"]
        def mean_abs(a, b):
            a = np.array(a, dtype=np.float32); b = np.array(b, dtype=np.float32)
            return float(np.abs(a - b).mean())
        rows.append(dict(
            sample_id=sid,
            suite=v["suite"],
            task_id=v["task_id"],
            base_l1=float(v["base_l1"]),
            visual_l1=float(v["visual_l1"]),
            delta_visual_base=float(delta_visual_base),
            visual_improves=bool(delta_visual_base < 0.0),
            delta_ao_base=float(delta_ao_base),
            delta_shuf_base=float(delta_shuf_base),
            residual_visual_base=mean_abs(v["visual_prediction"], v["base_prediction"]),
            ood_scene_k1=float(k1_scene[i]),
            ood_scene_k10=float(k10_scene[i]),
            ood_inst_k1=float(k1_inst[i]),
            ood_inst_k10=float(k10_inst[i]),
        ))

    out_rows_path = os.path.join(OUT, "diagnostic_rows.json")
    with open(out_rows_path, "w") as f:
        json.dump(rows, f, indent=1)

    # Aggregations.
    arr = np.array([r["delta_visual_base"] for r in rows], dtype=np.float64)
    imp = np.array([r["visual_improves"] for r in rows])
    ood = np.array([r["ood_scene_k1"] for r in rows])
    ood10 = np.array([r["ood_scene_k10"] for r in rows])
    base_l1 = np.array([r["base_l1"] for r in rows])
    resid = np.array([r["residual_visual_base"] for r in rows])

    qcut = np.quantile(ood, [0.25, 0.5, 0.75])
    quart = np.digitize(ood, qcut)

    def summarize_labels(labels_name, group):
        out = {}
        for q in sorted(set(group)):
            mask = group == q
            out[int(q)] = dict(n=int(mask.sum()),
                               improvement_rate=float(imp[mask].mean()),
                               mean_delta_l1=float(arr[mask].mean()))
        return out

    qscene = summarize_labels("scene quartile", quart)

    base_qcut = np.quantile(base_l1, [0.25, 0.5, 0.75])
    base_quart = np.digitize(base_l1, base_qcut)

    # Adequacy-stratified improvement (visual improvement over base by base adequacy).
    adequacy = {}
    for q in range(4):
        mask = base_quart == q
        adequacy[int(q)] = dict(n=int(mask.sum()),
                                improvement_rate=float(imp[mask].mean()),
                                mean_delta_l1=float(arr[mask].mean()),
                                mean_residual=float(resid[mask].mean()))

    # Already-adequate band (base_l1 < scalar that reproduces 144 samples).
    near = base_l1 < 0.02
    near_stats = dict(n=int(near.sum()),
                      improvement_rate=float(imp[near].mean()),
                      mean_delta_l1=float(arr[near].mean()),
                      mean_residual=float(resid[near].mean()))

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        rx -= rx.mean(); ry -= ry.mean()
        den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        return float((rx * ry).sum() / den) if den > 0 else float("nan")

    corr = dict(
        delta_vs_ood_k1=spearman(arr, ood),
        delta_vs_ood_k10=spearman(arr, ood10),
        delta_vs_base_l1=spearman(arr, base_l1),
        delta_vs_residual=spearman(arr, resid),
        residual_vs_base_l1=spearman(resid, base_l1),
    )

    # Per suite.
    from collections import defaultdict
    su = defaultdict(list)
    for r in rows:
        su[r["suite"]].append(r)
    per_suite = {}
    for s, sr in su.items():
        dela = np.array([x["delta_visual_base"] for x in sr])
        impa = np.array([x["visual_improves"] for x in sr])
        ooda = np.array([x["ood_scene_k1"] for x in sr])
        per_suite[s] = dict(n=len(sr),
                            improvement_rate=float(impa.mean()),
                            mean_delta_l1=float(dela.mean()),
                            median_ood_k1=float(np.median(ooda)))

    summary = dict(
        note="read-only offline diagnostic on saved 2026-09-16 collection and 2026-09-18 fit evidence",
        scene_ood_quartiles=qcut.tolist(),
        base_adequacy_quartiles=base_qcut.tolist(),
        scene_quartile=qscene,
        base_adequacy=adequacy,
        near_base_lt_002=near_stats,
        spearman=corr,
        per_suite=per_suite,
        overall=dict(n=800, improvement_rate=float(imp.mean()),
                     mean_delta_visual_base=float(arr.mean())),
    )
    out_sum = os.path.join(OUT, "diagnostic_summary.json")
    with open(out_sum, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()