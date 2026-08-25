#!/usr/bin/env python3
"""Execute the frozen CPU/OSMesa BRACE-B1 replay-equivalence gate."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_ROOT = Path("/home/ved/SAVR")
CONFIG_RELATIVE = Path("configs/brace/b1_replay_v1.json")
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(dict(value)) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


class ResourceBudget:
    def __init__(self, *, environments: int, steps: int) -> None:
        self.environment_cap = environments
        self.step_cap = steps
        self.environment_instances = 0
        self.simulator_steps = 0

    def environment(self) -> None:
        if self.environment_instances >= self.environment_cap:
            raise RuntimeError("B1 environment-instance cap exceeded")
        self.environment_instances += 1

    def step(self) -> None:
        if self.simulator_steps >= self.step_cap:
            raise RuntimeError("B1 simulator-step cap exceeded")
        self.simulator_steps += 1

    def as_record(self) -> dict[str, int]:
        return {
            "environment_instances": self.environment_instances,
            "environment_cap": self.environment_cap,
            "simulator_steps": self.simulator_steps,
            "simulator_step_cap": self.step_cap,
        }


def validate_config(config: Mapping[str, Any]) -> None:
    supplied = config.get("semantic_sha256")
    payload = dict(config)
    payload.pop("semantic_sha256", None)
    if semantic_sha256(payload) != supplied:
        raise RuntimeError("B1 configuration semantic hash mismatch")
    if config.get("schema_version") != "brace.b1-config.v1":
        raise RuntimeError("B1 configuration schema changed")
    if config.get("prefix_lengths") != [3, 6, 10]:
        raise RuntimeError("B1 prefix schedule changed")
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 3:
        raise RuntimeError("B1 requires exactly three scenarios")
    if [item.get("scenario_id") for item in scenarios] != [
        "free-motion",
        "contact",
        "gripper-transition",
    ]:
        raise RuntimeError("B1 scenario identities changed")
    if any(len(item.get("actions", [])) != 12 for item in scenarios):
        raise RuntimeError("B1 scenarios require exactly twelve actions")
    caps = config.get("resource_caps")
    if caps != {
        "cuda_visible": False,
        "model_queries": 0,
        "policy_outcomes": 0,
        "environment_instances": 30,
        "simulator_steps": 240,
        "wall_seconds": 1800,
        "artifact_bytes": 268435456,
        "downloads_allowed": False,
    }:
        raise RuntimeError("B1 resource caps changed")


def configure_process(root: Path) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("B1 refuses a visible CUDA device")
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "PYTHONNOUSERSITE": "1",
            "LIBERO_CONFIG_PATH": str(root / "cache/libero"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "TF_CPP_MIN_LOG_LEVEL": "3",
        }
    )


def numeric_field(value: Any, np: Any) -> dict[str, Any]:
    array = np.asarray(value)
    floating = np.asarray(array, dtype=np.float64)
    if not bool(np.isfinite(floating).all()):
        raise RuntimeError("B1 snapshot contains nonfinite numeric values")
    return {
        "kind": "numeric",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "values": floating.reshape(-1).tolist(),
        "raw_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def exact_array_field(value: Any, np: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "kind": "exact",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def contact_pairs(environment: Any) -> list[list[str]]:
    pairs: list[list[str]] = []
    for index in range(int(environment.sim.data.ncon)):
        contact = environment.sim.data.contact[index]
        first = environment.sim.model.geom_id2name(int(contact.geom1)) or f"geom-{contact.geom1}"
        second = environment.sim.model.geom_id2name(int(contact.geom2)) or f"geom-{contact.geom2}"
        pairs.append(sorted((str(first), str(second))))
    return sorted(pairs)


def snapshot(
    environment: Any,
    observation: Mapping[str, Any],
    *,
    reward: float,
    done: bool,
    next_action_index: int,
    np: Any,
) -> dict[str, Any]:
    images: dict[str, Any] = {}
    continuous: dict[str, Any] = {}
    for key in sorted(observation):
        value = observation[key]
        if key.endswith("_image"):
            images[key] = exact_array_field(value, np)
        elif key != "image-state":
            continuous[key] = numeric_field(value, np)

    robot = environment.robots[0]
    controller = robot.controller
    controller_fields: dict[str, Any] = {}
    for name in ("goal_pos", "goal_ori", "initial_ee_pos", "initial_ee_ori_mat"):
        if hasattr(controller, name):
            controller_fields[name] = numeric_field(getattr(controller, name), np)

    inner = environment.env
    result = {
        "snapshot_version": "brace.b1-snapshot.v1",
        "sim_state": numeric_field(environment.get_sim_state(), np),
        "observations": continuous,
        "images": images,
        "controller": controller_fields,
        "contacts": {
            "count": int(environment.sim.data.ncon),
            "pairs": contact_pairs(environment),
        },
        "counters": {
            "timestep": int(inner.timestep),
            "cur_time": float(inner.cur_time),
            "sim_time": float(environment.sim.data.time),
            "done": bool(inner.done),
        },
        "outcome": {
            "reward": float(reward),
            "success": bool(environment.check_success()),
            "done": bool(done),
        },
        "queue": {
            "executed_actions": int(next_action_index),
            "next_action_index": int(next_action_index),
        },
    }
    result["snapshot_sha256"] = semantic_sha256(result)
    return result


def vector_from_snapshot(record: Mapping[str, Any], key: str, np: Any) -> Any:
    field = record["observations"][key]
    return np.asarray(field["values"], dtype=np.float64).reshape(field["shape"])


def scenario_evidence(transcript: Mapping[str, Any], scenario: Mapping[str, Any], np: Any) -> dict[str, Any]:
    snapshots = [transcript["initial_snapshot"], *[step["snapshot"] for step in transcript["steps"]]]
    required = scenario["required_evidence"]
    if required == "eef-displacement":
        positions = [vector_from_snapshot(item, "robot0_eef_pos", np) for item in snapshots]
        initial = positions[0]
        maximum = max(float(np.linalg.norm(value - initial)) for value in positions[1:])
        return {"kind": required, "maximum_displacement": maximum, "passed": maximum >= 0.01}
    if required == "gripper-displacement":
        positions = [vector_from_snapshot(item, "robot0_gripper_qpos", np) for item in snapshots]
        flattened = np.concatenate([value.reshape(-1) for value in positions])
        span = float(np.max(flattened) - np.min(flattened))
        return {"kind": required, "joint_position_span": span, "passed": span >= 0.001}
    if required == "contact-signature":
        initial = {tuple(pair) for pair in snapshots[0]["contacts"]["pairs"]}
        later = {tuple(pair) for item in snapshots[1:] for pair in item["contacts"]["pairs"]}
        novel = sorted([list(pair) for pair in later - initial])
        return {
            "kind": required,
            "initial_contact_pairs": len(initial),
            "novel_contact_pairs": novel,
            "passed": bool(novel),
        }
    raise RuntimeError(f"Unknown B1 evidence requirement: {required}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run B1 outside {EXPECTED_ROOT}: {root}")
    configure_process(root)
    config = json.loads((root / CONFIG_RELATIVE).read_text())
    validate_config(config)
    caps = config["resource_caps"]

    def timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError("BRACE-B1 reached its frozen wall cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(int(caps["wall_seconds"]))

    if git_output(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Refusing B1 with modified tracked project files")
    source_revision = git_output(root, "rev-parse", "HEAD")
    libero_root = root / "third_party/LIBERO"
    libero_revision = git_output(libero_root, "rev-parse", "HEAD")
    if libero_revision != LIBERO_REVISION or git_output(libero_root, "status", "--porcelain"):
        raise SystemExit("Pinned LIBERO source is missing, changed, or dirty")

    run_root = root / "results" / config["run_id"]
    if run_root.exists():
        raise SystemExit(f"Immutable B1 run already exists: {run_root}")
    run_root.mkdir(parents=True)
    manifest = {
        "schema_version": "brace.b1-manifest.v1",
        "run_id": config["run_id"],
        "status": "running",
        "started_at_utc": utc_now(),
        "source_revision": source_revision,
        "libero_revision": libero_revision,
        "configuration_sha256": file_sha256(root / CONFIG_RELATIVE),
        "configuration_semantic_sha256": config["semantic_sha256"],
        "environment": {
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "mujoco_gl": os.environ["MUJOCO_GL"],
            "pyopengl_platform": os.environ["PYOPENGL_PLATFORM"],
            "libero_config_path": os.environ["LIBERO_CONFIG_PATH"],
        },
        "resource_caps": caps,
        "command": "envs/openvla-oft/bin/python scripts/run_brace_b1.py",
    }
    write_once(run_root / "manifest.json", manifest)

    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(libero_root))
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from savr.brace.b1 import freeze_transcript, validate_reconstruction, validate_transcript

    budget = ResourceBudget(
        environments=int(caps["environment_instances"]),
        steps=int(caps["simulator_steps"]),
    )
    task_suite = benchmark.get_benchmark_dict()[config["suite"]]()
    initial_states: dict[int, Any] = {}

    def create_environment(task_id: int) -> tuple[Any, Mapping[str, Any], Any]:
        budget.environment()
        task = task_suite.get_task(task_id)
        bddl_path = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        environment = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=int(config["camera_resolution"]),
            camera_widths=int(config["camera_resolution"]),
            horizon=100,
        )
        environment.seed(int(config["seed"]))
        environment.reset()
        if task_id not in initial_states:
            initial_states[task_id] = task_suite.get_task_init_states(task_id)[
                int(config["initial_state_id"])
            ].copy()
        observation = environment.set_init_state(initial_states[task_id].copy())
        return environment, observation, task

    def take_step(environment: Any, action: Sequence[float]) -> tuple[Mapping[str, Any], float, bool]:
        budget.step()
        observation, reward, done, _ = environment.step(np.asarray(action, dtype=np.float32))
        return observation, float(reward), bool(done)

    results: list[dict[str, Any]] = []
    started = time.monotonic()
    terminal_error: BaseException | None = None
    try:
        for scenario in config["scenarios"]:
            scenario_id = scenario["scenario_id"]
            task_id = int(scenario["task_id"])
            actions = [[float(value) for value in action] for action in scenario["actions"]]
            environment, observation, task = create_environment(task_id)
            try:
                initial = snapshot(
                    environment,
                    observation,
                    reward=0.0,
                    done=False,
                    next_action_index=0,
                    np=np,
                )
                step_snapshots: list[dict[str, Any]] = []
                for index, action in enumerate(actions):
                    observation, reward, done = take_step(environment, action)
                    step_snapshots.append(
                        snapshot(
                            environment,
                            observation,
                            reward=reward,
                            done=done,
                            next_action_index=index + 1,
                            np=np,
                        )
                    )
                transcript = freeze_transcript(
                    metadata={
                        "scenario_id": scenario_id,
                        "suite": config["suite"],
                        "task_id": task_id,
                        "task_name": task.name,
                        "language": task.language,
                        "initial_state_id": config["initial_state_id"],
                        "seed": config["seed"],
                        "configuration_semantic_sha256": config["semantic_sha256"],
                        "source_revision": source_revision,
                        "libero_revision": libero_revision,
                    },
                    initial_snapshot=initial,
                    actions=actions,
                    step_snapshots=step_snapshots,
                )
                validate_transcript(transcript)
                write_once(run_root / "transcripts" / f"{scenario_id}.json", transcript)
            finally:
                environment.close()

            evidence = scenario_evidence(transcript, scenario, np)
            if not evidence["passed"]:
                raise RuntimeError(f"{scenario_id} did not produce required {evidence['kind']} evidence")

            prefix_results: list[dict[str, Any]] = []
            for prefix_length in config["prefix_lengths"]:
                replay_snapshots: list[dict[str, Any]] = []
                probe_snapshots: list[dict[str, Any]] = []
                replay_verdicts: list[dict[str, Any]] = []
                for repetition in range(2):
                    environment, observation, _task = create_environment(task_id)
                    try:
                        reward, done = 0.0, False
                        for action in actions[:prefix_length]:
                            observation, reward, done = take_step(environment, action)
                        reconstructed = snapshot(
                            environment,
                            observation,
                            reward=reward,
                            done=done,
                            next_action_index=prefix_length,
                            np=np,
                        )
                        replay_snapshots.append(reconstructed)
                        verdict = validate_reconstruction(
                            transcript["steps"][prefix_length - 1]["snapshot"],
                            reconstructed,
                            restoration_mode="env_step_prefix",
                            absolute_tolerance=float(config["continuous_absolute_tolerance"]),
                            relative_tolerance=float(config["continuous_relative_tolerance"]),
                        )
                        replay_verdicts.append(verdict.as_record())
                        if not verdict.accepted:
                            raise RuntimeError(
                                f"{scenario_id} prefix {prefix_length} replay {repetition} diverged"
                            )
                        probe = [*config["probe_motion"], actions[prefix_length - 1][-1]]
                        observation, reward, done = take_step(environment, probe)
                        probe_snapshots.append(
                            snapshot(
                                environment,
                                observation,
                                reward=reward,
                                done=done,
                                next_action_index=prefix_length + 1,
                                np=np,
                            )
                        )
                    finally:
                        environment.close()
                probe_verdict = validate_reconstruction(
                    probe_snapshots[0],
                    probe_snapshots[1],
                    restoration_mode="env_step_prefix",
                    absolute_tolerance=float(config["continuous_absolute_tolerance"]),
                    relative_tolerance=float(config["continuous_relative_tolerance"]),
                )
                if not probe_verdict.accepted:
                    raise RuntimeError(f"{scenario_id} prefix {prefix_length} probe diverged")
                prefix_results.append(
                    {
                        "prefix_length": prefix_length,
                        "replays": replay_verdicts,
                        "probe": probe_verdict.as_record(),
                        "probe_snapshot_sha256": [item["snapshot_sha256"] for item in probe_snapshots],
                    }
                )

            negative_prefix = max(config["prefix_lengths"])
            modified_actions = [list(action) for action in actions]
            modified_actions[int(config["modified_action_index"])][
                int(config["modified_action_dimension"])
            ] += float(config["modified_action_delta"])
            environment, observation, _task = create_environment(task_id)
            try:
                reward, done = 0.0, False
                for action in modified_actions[:negative_prefix]:
                    observation, reward, done = take_step(environment, action)
                modified_snapshot = snapshot(
                    environment,
                    observation,
                    reward=reward,
                    done=done,
                    next_action_index=negative_prefix,
                    np=np,
                )
            finally:
                environment.close()
            modified_verdict = validate_reconstruction(
                transcript["steps"][negative_prefix - 1]["snapshot"],
                modified_snapshot,
                restoration_mode="env_step_prefix",
                absolute_tolerance=float(config["continuous_absolute_tolerance"]),
                relative_tolerance=float(config["continuous_relative_tolerance"]),
            )
            if modified_verdict.accepted:
                raise RuntimeError(f"{scenario_id} modified-prefix negative control was accepted")

            target_state = np.asarray(
                transcript["steps"][negative_prefix - 1]["snapshot"]["sim_state"]["values"],
                dtype=np.float64,
            )
            environment, _observation, _task = create_environment(task_id)
            try:
                direct_observation = environment.regenerate_obs_from_state(target_state)
                direct_snapshot = snapshot(
                    environment,
                    direct_observation,
                    reward=float(environment.env.reward()),
                    done=bool(environment.env.done),
                    next_action_index=negative_prefix,
                    np=np,
                )
            finally:
                environment.close()
            direct_verdict = validate_reconstruction(
                transcript["steps"][negative_prefix - 1]["snapshot"],
                direct_snapshot,
                restoration_mode="direct_state_only",
                absolute_tolerance=float(config["continuous_absolute_tolerance"]),
                relative_tolerance=float(config["continuous_relative_tolerance"]),
            )
            structural = [
                item for item in direct_verdict.mismatches if not item.startswith("restoration_mode:")
            ]
            if direct_verdict.accepted or not structural:
                raise RuntimeError(f"{scenario_id} direct-state negative control lacked structural rejection")

            scenario_result = {
                "scenario_id": scenario_id,
                "task_id": task_id,
                "transcript_sha256": transcript["transcript_sha256"],
                "required_evidence": evidence,
                "prefix_checks": prefix_results,
                "modified_prefix_control": modified_verdict.as_record(),
                "direct_state_control": direct_verdict.as_record(),
                "direct_state_structural_mismatches": structural,
                "passed": True,
            }
            write_once(run_root / "checks" / f"{scenario_id}.json", scenario_result)
            results.append(scenario_result)

        expected_environments = 27
        expected_steps = 198
        if budget.environment_instances != expected_environments or budget.simulator_steps != expected_steps:
            raise RuntimeError(
                f"B1 accounting mismatch: {budget.environment_instances} envs, "
                f"{budget.simulator_steps} steps"
            )
        if directory_size(run_root) > int(caps["artifact_bytes"]):
            raise RuntimeError("B1 artifact cap exceeded")
        summary = {
            "schema_version": "brace.b1-summary.v1",
            "run_id": config["run_id"],
            "status": "accepted",
            "started_at_utc": manifest["started_at_utc"],
            "completed_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "source_revision": source_revision,
            "libero_revision": libero_revision,
            "configuration_semantic_sha256": config["semantic_sha256"],
            "scenario_count": len(results),
            "transcript_count": len(results),
            "positive_prefix_replays": sum(
                len(prefix["replays"])
                for result in results
                for prefix in result["prefix_checks"]
            ),
            "identical_probe_checks": sum(len(result["prefix_checks"]) for result in results),
            "modified_prefix_rejections": sum(
                not result["modified_prefix_control"]["accepted"] for result in results
            ),
            "direct_state_rejections": sum(
                not result["direct_state_control"]["accepted"] for result in results
            ),
            "required_scenario_evidence": [result["required_evidence"] for result in results],
            "resource_usage": budget.as_record(),
            "artifact_bytes_before_summary": directory_size(run_root),
            "gates": {
                "three_scenarios": len(results) == 3,
                "all_transcripts_valid": all(result["passed"] for result in results),
                "all_positive_replays_equal": all(
                    replay["accepted"]
                    for result in results
                    for prefix in result["prefix_checks"]
                    for replay in prefix["replays"]
                ),
                "all_probe_transitions_equal": all(
                    prefix["probe"]["accepted"]
                    for result in results
                    for prefix in result["prefix_checks"]
                ),
                "all_modified_prefixes_rejected": all(
                    not result["modified_prefix_control"]["accepted"] for result in results
                ),
                "all_direct_state_controls_rejected": all(
                    not result["direct_state_control"]["accepted"] for result in results
                ),
                "all_scenario_evidence_present": all(
                    result["required_evidence"]["passed"] for result in results
                ),
                "resource_caps_respected": True,
                "cuda_hidden": os.environ["CUDA_VISIBLE_DEVICES"] == "",
                "no_model_or_policy_queries": True,
            },
        }
        summary["all_gates_passed"] = all(summary["gates"].values())
        if not summary["all_gates_passed"]:
            raise RuntimeError("B1 summary gates did not all pass")
        summary["semantic_sha256"] = semantic_sha256(summary)
        write_once(run_root / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        terminal_error = error
        failure = {
            "schema_version": "brace.b1-technical-stop.v1",
            "run_id": config["run_id"],
            "status": "technical_stop",
            "recorded_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "source_revision": source_revision,
            "configuration_semantic_sha256": config["semantic_sha256"],
            "resource_usage": budget.as_record(),
            "completed_scenarios": [item["scenario_id"] for item in results],
        }
        failure["semantic_sha256"] = semantic_sha256(failure)
        write_once(run_root / "technical_stop.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        signal.alarm(0)
        if terminal_error is not None:
            sys.stderr.flush()


if __name__ == "__main__":
    raise SystemExit(main())
