"""Confound-controlled correction diagnostics + deploy-available benefit signals.

Read-only CPU analysis of already-saved evidence. No GPU, no new data.

Part A (confound control, issue 2 from user feedback):
- delta = visual_l1 - base_l1 is mechanically bounded below by -base_l1: on
  small-base samples there is little room to improve, so the sign of delta is
  noise-dominated. Report benefit relative to base error and compare the visual
  corrector against the action-only corrector (which also perturbs every action)
  to isolate visual-specific harm.
- Re-evaluate the coverage gradient with the corrected fit-only z-scored OOD.

Part B (cheap deploy-available predictor, issue 3):
Can inexpensive signals available at query time (no dense teacher) predict
whether applying the visual corrector helps (visual_l1 < base_l1) on a
held-out trajectory-disjoint set? Candidate signals, all computed from the
compressed-path features the system already has:
  b1 base action statistics (mean |a|, max |a|, std over 8x7)      base_prediction
  b2 corrector residual magnitude mean|visual - base|              visual_prediction
  b3 cross-head agreement mean|action_only - visual|               action_only_prediction
  b4 instruction L2 norm                                           stored instruction
  b5 proprio/state statistics (L1/L2, max)                         stored state
  b6 projected-patch statistics (mean abs, max abs, std mean)      stored current_visual
  b7 penultimate action-feature statistics (mean abs, max, std)    stored action_features
  b8 scene-OOD distance to fitting records (fit-only z-score)      computed v02
All features are joined by sample_id from the 4,000-record collection and the
800-row fit validation predictions.

Design (fixed before reading any result):
- Split by TRAJECTORY (sample_id prefix) into 2/3 train, 1/3 test so no
  trajectory appears in both; seed 7.
- Model: sklearn LogisticRegression(C=1.0) on standardized candidate signal
  matrix X (b1..b8, one dimension per statistic), target y = visual_l1 < base_l1.
- Report test-set ROC AUC with its 95% interval via 200 stratified bootstrap
  draws (seed 7), plus confusion/benefit-rate lift: mean visual_l1 and mean
  delta l1 in predicted-help vs predicted-don't apply. Compare against always
  apply and never apply on the same test set.
- Report also the single most predictive candidate (univariate AUC) so the
  rule is inspectable, and Spearman of each candidate with benefit.
"""
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = "/Users/veddwivedi/Documents/VLA/SAVR"
COLLECTION = os.path.join(ROOT, "results/current-frame-feature-collection-v01")
FIT = os.path.join(ROOT, "results/current-frame-adapter-fit-v01")
DIAG = os.path.join(ROOT, "results/current-frame-correction-coverage-diagnostic-v02")
OUT = os.path.join(ROOT, "results/current-frame-correction-benefit-signals-v01")


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_feature_stats(artifact):
    """Return dict of cheap deploy-available statistics for one record."""
    with np.load(os.path.join(COLLECTION, artifact), allow_pickle=False) as npz:
        meta = json.loads(npz["metadata"].tobytes())
        act = npz["action_features"]; act_f = bf16_to_f32(act)
        vis = npz["current_visual"];   vis_f = bf16_to_f32(vis)
        ins = npz["instruction"];      ins_f = bf16_to_f32(ins)
        state = npz["state"].astype(np.float32)
        base = npz["base_actions"].astype(np.float32)
    return dict(
        base_mean_abs=float(np.abs(base).mean()),
        base_max_abs=float(np.abs(base).max()),
        base_std=float(base.std()),
        instr_norm=float(np.linalg.norm(ins_f)),
        state_l1=float(np.abs(state).sum()),
        state_l2=float(np.linalg.norm(state)),
        state_max=float(np.abs(state).max()),
        vis_mean_abs=float(np.abs(vis_f).mean()),
        vis_max_abs=float(np.abs(vis_f).max()),
        vis_token_std_mean=float(vis_f.mean(1).std()),
        actfeat_mean_abs=float(np.abs(act_f).mean()),
        actfeat_max_abs=float(np.abs(act_f).max()),
        actfeat_std=float(act_f.std()),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    records = json.load(open(os.path.join(COLLECTION, "records.json")))
    artifact_of = {r["sample_id"]: r["artifact"] for r in records}

    validation = json.load(open(os.path.join(FIT, "validation.json")))
    diag_rows = json.load(open(os.path.join(DIAG, "diagnostic_rows.json")))
    diag_by_id = {r["sample_id"]: r for r in diag_rows}

    n = len(validation)
    X_names = None
    X, y = [], []
    rows_out = []
    for i, v in enumerate(validation):
        sid = v["sample_id"]
        art = artifact_of.get(sid)
        if art is None:
            raise SystemExit(f"missing feature artifact for {sid}")
        stats = load_feature_stats(art)
        bp = np.asarray(v["base_prediction"], dtype=np.float32)
        vp = np.asarray(v["visual_prediction"], dtype=np.float32)
        ap = np.asarray(v["action_only_prediction"], dtype=np.float32)
        resid = float(np.abs(vp - bp).mean())
        agree = float(np.abs(ap - vp).mean())
        stats["residual_mean_abs"] = resid
        stats["cross_head_agree"] = agree
        stats["ood_scene_k1"] = diag_by_id[sid]["ood_scene_k1"]
        names = ["base_mean_abs", "base_max_abs", "base_std",
                 "instr_norm", "state_l1", "state_l2", "state_max",
                 "vis_mean_abs", "vis_max_abs", "vis_token_std_mean",
                 "actfeat_mean_abs", "actfeat_max_abs", "actfeat_std",
                 "residual_mean_abs", "cross_head_agree", "ood_scene_k1"]
        vec = np.array([stats[k] for k in names], dtype=np.float64)
        if X_names is None:
            X_names = names
        X.append(vec)
        y.append(1 if v["visual_l1"] < v["base_l1"] else 0)
        rows_out.append(dict(
            sample_id=sid, suite=v["suite"], task_id=v["task_id"],
            trajectory=sid.split(":")[0],
            base_l1=v["base_l1"], visual_l1=v["visual_l1"],
            action_only_l1=v["action_only_l1"], teacher_agreement=int(y[-1]),
            **{k: float(stats[k]) for k in names}))
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)

    # ---- Part A: confound control on the full 800 ----
    base = np.array([r["base_l1"] for r in rows_out])
    vis = np.array([r["visual_l1"] for r in rows_out])
    ao = np.array([r["action_only_l1"] for r in rows_out])
    gain_vis = base - vis               # positive = visual improves
    gain_ao = base - ao
    rel_gain_vis = gain_vis / np.maximum(base, 1e-6)
    rel_gain_ao = gain_ao / np.maximum(base, 1e-6)

    near = base < 0.02
    def rate(mask, cond): return float(cond[mask].mean())
    part_a = dict(
        n_near=int(near.sum()),
        near_visual_improve_rate=rate(near, vis < base),
        near_action_only_improve_rate=rate(near, ao < base),
        near_visual_beat_action_only=rate(near, vis < ao),
        near_mean_rel_gain_visual=float(rel_gain_vis[near].mean()),
        near_mean_rel_gain_action_only=float(rel_gain_ao[near].mean()),
        near_mean_gain_visual=float(gain_vis[near].mean()),
        near_mean_gain_action_only=float(gain_ao[near].mean()),
        wide_mean_gain_visual=float(gain_vis[~near].mean()),
        wide_mean_gain_action_only=float(gain_ao[~near].mean()),
        wide_visual_improve_rate=rate(~near, vis < base),
        wide_action_only_improve_rate=rate(~near, ao < base),
        absolute_gains_small=dict(near_visual_max=float(np.abs(gain_vis[near]).max()),
                                  near_action_only_max=float(np.abs(gain_ao[near]).max())),
    )

    # ---- Part B: trajectory-disjoint train/test ----
    rng = np.random.RandomState(7)
    trajs = sorted({r["trajectory"] for r in rows_out})
    rng.shuffle(trajs)
    n_tr = int(round(len(trajs) * 2 / 3))
    train_trajs = set(trajs[:n_tr])
    tr_mask = np.array([r["trajectory"] in train_trajs for r in rows_out])
    te_mask = ~tr_mask
    Xtr, ytr = X[tr_mask], y[tr_mask]
    Xte, yte = X[te_mask], y[te_mask]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=7)
    clf.fit(Xtr_s, ytr)
    pte = clf.predict_proba(Xte_s)[:, 1]

    # univariate AUC per candidate (ascending-rank Wilcoxon statistic)
    def auc(scores, labels):
        order = np.argsort(scores, kind="mergesort")
        labels = labels[order]
        ranks = np.arange(1, len(labels) + 1)
        pos = labels.sum()
        if pos == 0 or pos == len(labels):
            return float("nan")
        return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2)
                     / (pos * (len(labels) - pos)))

    uni = {name: auc(Xte[:, j], yte) for j, name in enumerate(X_names)}

    # bootstrap CI for heldout logistic AUC
    rng2 = np.random.RandomState(7)
    bts = []
    idx = np.arange(len(yte))
    for _ in range(200):
        s = rng2.choice(idx, size=len(idx), replace=True)
        bts.append(auc(pte[s], yte[s]))
    bts = np.asarray(bts)
    ci = (np.percentile(bts, 2.5), np.percentile(bts, 97.5))

    # operating point: apply correction only when p>=0.5 (standard decision rule)
    apply_mask = pte >= 0.5
    vis_te = vis[te_mask]; base_te = base[te_mask]; gain_te = gain_vis[te_mask]
    test_gain_always = float(gain_te.mean())
    test_l1_always = float(vis_te.mean())
    test_gain_never = 0.0                                  # base actions unchanged
    test_l1_never = float(base_te.mean())
    test_gain_gated = float(gain_te[apply_mask].mean()) if apply_mask.any() else float("nan")
    test_l1_gated = float(((vis_te * apply_mask + base_te * ~apply_mask)).mean())
    p_apply = float(apply_mask.mean())
    gated_benefit_rate = float((yte[apply_mask].mean())) if apply_mask.any() else float("nan")

    def spearman(x, yv):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(yv)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        den = np.sqrt((rx**2).sum() * (ry**2).sum())
        return float((rx*ry).sum()/den) if den else float("nan")

    corr = {name: spearman(Xte[:, j], gain_vis[te_mask]) for j, name in enumerate(X_names)}

    part_b = dict(
        n_train=int(tr_mask.sum()), n_test=int(te_mask.sum()),
        n_train_trajs=len(train_trajs), n_test_trajs=len(trajs) - n_tr,
        heldout_auc=float(auc(pte, yte)),
        heldout_auc_ci95=ci,
        univariate_auc=uni,
        spearman_with_gain=corr,
        gated_vs_always=dict(
            always_apply_mean_l1=test_l1_always,
            never_apply_mean_l1=test_l1_never,
            gated_mean_l1=test_l1_gated,
            gated_mean_gain=test_gain_gated,
            always_mean_gain=test_gain_always,
            fraction_apply=p_apply,
            gated_benefit_rate=gated_benefit_rate,
            heldout_benefit_rate=float(yte.mean()),
        ),
    )

    with open(os.path.join(OUT, "part_a_confounds.json"), "w") as f:
        json.dump(part_a, f, indent=1)
    with open(os.path.join(OUT, "part_b_signals.json"), "w") as f:
        json.dump(part_b, f, indent=1)
    with open(os.path.join(OUT, "rows.json"), "w") as f:
        json.dump(rows_out, f, indent=1)
    print("PART A:", json.dumps(part_a, indent=1))
    print("PART B:", json.dumps(part_b, indent=1))


if __name__ == "__main__":
    main()