"""Raw authenticated demonstration inputs; distinct from live 224-pixel images."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class OfflineCameraPair:
    step: int
    scene: np.ndarray
    wrist: np.ndarray

    @classmethod
    def capture(cls, observation, step):
        if type(step) is not int or step not in (0, 1):
            raise ValueError("only frozen offline frames 0 and 1 are allowed")
        images = []
        for key in ("full_image", "wrist_image"):
            value = np.asarray(observation[key])
            if value.shape != (128, 128, 3) or value.dtype != np.uint8:
                raise ValueError("raw demonstration image must be uint8 RGB 128x128")
            value = value.copy()
            value.flags.writeable = False
            images.append(value)
        return cls(step, *images)


def observation_sha256(obs):
    digest = hashlib.sha256()
    for name in ("full_image", "wrist_image", "state"):
        value = np.asarray(obs[name])
        digest.update(name.encode()); digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_offline_inputs(root, manifest):
    """CPU-only, validate actual values before loading any model. No action labels."""
    import h5py
    root = root.resolve()
    inputs = manifest["inputs"]
    if len(inputs) != 8 or len({r["trajectory_id"] for r in inputs}) != 8:
        raise ValueError("exactly eight unique consumed training trajectories required")
    observations, audit = {}, []
    for row in inputs:
        path = (root / manifest["data_root_relative"] / row["source_path"]).resolve()
        if not path.is_relative_to(root) or row["split"] != "train":
            raise ValueError("input must be inside project and training split")
        if not row["language_instruction"] or not row["normalization_statistics_key"]:
            raise ValueError("instruction and normalization identity required")
        with h5py.File(path, "r") as handle:
            group = handle[f"data/{row['original_trajectory_id']}/obs"]
            expected = dict(agentview_rgb=((128, 128, 3), np.uint8),
                            eye_in_hand_rgb=((128, 128, 3), np.uint8),
                            ee_states=((6,), np.float64), gripper_states=((2,), np.float64))
            for name, (shape, dtype) in expected.items():
                field = group[name]
                if field.shape != (row["step_count"], *shape) or field.dtype != dtype or field.shape[0] < 2:
                    raise ValueError(f"offline field schema differs: {name}")
            frames = []
            for frame in (0, 1):
                obs = dict(full_image=np.asarray(group["agentview_rgb"][frame]).copy(),
                           wrist_image=np.asarray(group["eye_in_hand_rgb"][frame]).copy(),
                           state=np.concatenate((group["ee_states"][frame], group["gripper_states"][frame])).copy())
                OfflineCameraPair.capture(obs, frame)
                if obs["state"].shape != (8,) or not np.isfinite(obs["state"]).all():
                    raise ValueError("invalid raw eight-dimensional robot state")
                audit.append(dict(trajectory_id=row["trajectory_id"], frame=frame,
                                  observation_sha256=observation_sha256(obs), raw_image_size=128))
                frames.append(obs)
        observations[row["trajectory_id"]] = (row, frames)
    return observations, audit


def audit_policy_preprocessing(observations, prepare_images, cfg):
    """Exercise the actual current/history wrapper and released preprocessing on CPU."""
    from .specprune_episode import prepare_query_camera_pairs
    checks = []
    for identity, frames in observations.values():
        previous = OfflineCameraPair.capture(frames[0], 0)
        for frame, observation in enumerate(frames):
            before = observation_sha256(observation)
            previous_bytes = (previous.scene.tobytes(), previous.wrist.tobytes())
            calls = []
            def traced(images, config):
                calls.append([np.asarray(im).shape for im in images])
                return prepare_images(images, config)
            current, past = prepare_query_camera_pairs(observation, previous, traced, cfg)
            expected_current = prepare_images([observation["full_image"].copy(), observation["wrist_image"].copy()], cfg)
            expected_past = prepare_images([previous.scene.copy(), previous.wrist.copy()], cfg)
            if calls != [[(128, 128, 3)] * 2] * 2:
                raise ValueError("raw inputs must enter released preprocessing exactly once per view")
            if any(np.asarray(a).tobytes() != np.asarray(b).tobytes()
                   for a, b in zip([*current, *past], [*expected_current, *expected_past])):
                raise ValueError("wrapped preprocessing differs from released raw-image preprocessing")
            if before != observation_sha256(observation) or previous_bytes != (previous.scene.tobytes(), previous.wrist.tobytes()):
                raise ValueError("preprocessing mutated raw observations")
            checks.append(dict(trajectory_id=identity["trajectory_id"], frame=frame,
                               released_preprocessing_equal=True, current_and_history_prepared_once=True,
                               observation_unchanged=True, output_image_size=224))
    return checks
