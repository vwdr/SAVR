import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from savr.openvla.contemporary_contract import validate_design, timing_slots, reconcile
from analyze_contemporary_reference import analyze, load_bundle


def fixture():
    design = json.loads((ROOT / "configs/openvla/contemporary_reference_design_v1.json").read_text())
    reference = json.loads((ROOT / design["reference_config"]).read_text())
    ids = [str(i) for i in range(8)]
    records = dict(episodes=[], timing=[], parity=[])
    for slot in design["episode_slots"]:
        records["episodes"].append(dict(slot_id=slot["slot_id"], condition_id=slot["condition"]["condition_id"],
            arm=slot["arm"], suite=slot["condition"]["suite"], success=True, policy_queries=1,
            executed_steps=1, replans=0, discarded_actions=0, remaining_actions=7,
            controller_enabled=slot["controller_enabled"], episode_seconds=2., simulator_seconds=.3,
            controller_seconds=.1, nonquery_preparation_seconds=0.,
            queries=[dict(query=1, seconds=.9, retained_visual_tokens=59 if slot["arm"] == "specprune" else 512, precise=False)]))
        if slot["arm"] == "native":
            records["parity"].append(dict(slot_id=slot["slot_id"], query=1, command_bytes_equal=True,
                errors=dict(hidden=0., normalized=0., raw_actions=0.), layers=[32, 32],
                observation_sha256="a" * 64, command_sha256="b" * 64))
    for row in timing_slots(ids):
        records["timing"].append(dict(row, seconds=1. if row["arm"] == "dense" else .8,
                                     query=row["frame"] + 1, precise=False,
                                     retained_visual_tokens=512 if row["arm"] == "dense" else 59))
    summary = dict(complete=True, automatic_retry=False, training_performed=False, positive_method_result=False,
        checkpoint_unchanged=True, authenticated_files_unchanged=True, elapsed_seconds=300.,
        peak_aggregate_gpu_memory_mib=16000, episodes=88, model_queries=232)
    return design, reference, summary, records, ids


class ContractTests(unittest.TestCase):
    def test_schedule_exact_and_balanced(self):
        d, r, *_ = fixture()
        self.assertTrue(validate_design(d, r))
        for offset in range(0, 88, 22):
            primary = d["episode_slots"][offset + 1:offset + 21]
            self.assertEqual(sum(primary[i]["arm"] == "dense" for i in range(0, 20, 2)), 5)

    def test_reject_mutated_schedules(self):
        for mutation in (lambda d: d["episode_slots"].pop(),
                         lambda d: d["episode_slots"].reverse(),
                         lambda d: d["episode_slots"][0].update(shadow_dense=False),
                         lambda d: d["episode_slots"][1]["condition"].update(initial_state_id=1),
                         lambda d: d["caps"].update(model_calls=6001),
                         lambda d: d["timing"].update(rounds=3)):
            d, r, *_ = fixture(); mutation(d)
            with self.assertRaises(ValueError): validate_design(d, r)

    def test_timing_count_reset_and_balance(self):
        rows = timing_slots([str(i) for i in range(8)])
        self.assertEqual(len(rows), 136)
        self.assertEqual(sum(r["warmup"] for r in rows), 8)
        for i in range(0, 136, 2):
            self.assertEqual([r["frame"] for r in rows[i:i+2]], [0, 1])
        for r in range(4):
            first = [q for q in rows if not q["warmup"] and q["round"] == r and q["frame"] == 0][::2]
            self.assertEqual(sum(q["arm"] == "dense" for q in first), 4)
        with self.assertRaises(ValueError): timing_slots(["duplicate"] * 8)

    def test_complete_fixture_and_no_success_gate(self):
        args = fixture()
        self.assertTrue(reconcile(*args))
        for row in args[3]["episodes"]: row["success"] = False
        self.assertTrue(reconcile(*args))

    def test_reject_corrupt_summary(self):
        for name, value in (("complete", False), ("episodes", 87), ("model_queries", 231),
                            ("elapsed_seconds", float("nan")), ("elapsed_seconds", 21600),
                            ("peak_aggregate_gpu_memory_mib", 23552), ("automatic_retry", True)):
            args = fixture(); args[2][name] = value
            with self.assertRaises(ValueError): reconcile(*args)

    def test_reject_missing_or_duplicate_records(self):
        for key in ("episodes", "timing", "parity"):
            for duplicate in (False, True):
                args = fixture()
                if duplicate: args[3][key][-1] = args[3][key][0]
                else: args[3][key].pop()
                with self.assertRaises(ValueError): reconcile(*args)

    def test_reject_action_and_timing_accounting(self):
        for name, value in (("remaining_actions", 8), ("discarded_actions", 1), ("replans", -1),
                            ("success", 1), ("executed_steps", 999), ("episode_seconds", .1)):
            args = fixture(); args[3]["episodes"][0][name] = value
            with self.assertRaises(ValueError): reconcile(*args)
        for value in (float("nan"), float("inf"), 0, -1):
            args = fixture(); args[3]["timing"][0]["seconds"] = value
            with self.assertRaises(ValueError): reconcile(*args)

    def test_reject_parity_mismatch(self):
        for mutation in (lambda p: p.update(command_bytes_equal=False),
                         lambda p: p["errors"].update(hidden=1.01e-6),
                         lambda p: p["errors"].update(normalized=float("nan")),
                         lambda p: p.update(layers=[31, 32]),
                         lambda p: p.update(command_sha256="z" * 64)):
            args = fixture(); mutation(args[3]["parity"][0])
            with self.assertRaises(ValueError): reconcile(*args)

    def test_analysis_separates_native_and_marks_discordance(self):
        args = fixture()
        # Only native-after fails; primary denominators must stay 40.
        args[3]["episodes"][21]["success"] = False
        result = analyze(*args, resamples=100)
        self.assertEqual((result["dense_successes"], result["specprune_successes"]), (40, 40))
        self.assertTrue(result["suites"][0]["repeatability_limited"])
        self.assertEqual(len(result["pairs"]), 40)
        self.assertAlmostEqual(result["controlled_trace_mean_time_reduction"], .2)
        self.assertFalse(result["positive_method_result"])

    def test_bad_comparator_outcomes_are_reported_not_rejected(self):
        args = fixture()
        for e in args[3]["episodes"]:
            if e["arm"] == "specprune": e["success"] = False
        result = analyze(*args, resamples=100)
        self.assertEqual(result["dense_only_successes"], 40)
        self.assertEqual(result["success_difference"], -1.)
        self.assertEqual(result["descriptive_paired_success_interval_95"], [-1., -1.])

    def test_stopped_run_cannot_be_analyzed(self):
        # Never open old raw records; the absence of a completed summary rejects first.
        with self.assertRaises(ValueError):
            load_bundle(ROOT, ROOT / "results/dense-precision-trace-v01")


if __name__ == "__main__": unittest.main()
