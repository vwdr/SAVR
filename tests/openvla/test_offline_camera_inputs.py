import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import h5py
import numpy as np
from savr.openvla.offline_camera_inputs import OfflineCameraPair, load_offline_inputs, audit_policy_preprocessing
from savr.openvla.specprune_episode import CameraPair, prepare_query_camera_pairs

ROOT = Path(__file__).resolve().parents[2]


def observation(size=128):
    return dict(full_image=np.full((size, size, 3), 21, np.uint8),
                wrist_image=np.full((size, size, 3), 73, np.uint8), state=np.zeros(8))


class OfflineCameraTests(unittest.TestCase):
    def test_raw_copy_preserves_pixels_and_blocks_mutation(self):
        obs = observation(); pair = OfflineCameraPair.capture(obs, 0)
        self.assertEqual(pair.scene.shape, (128, 128, 3))
        self.assertEqual(pair.scene.tobytes(), obs['full_image'].tobytes())
        self.assertFalse(np.shares_memory(pair.scene, obs['full_image']))
        with self.assertRaises(ValueError): pair.scene[0, 0, 0] = 0
        obs['full_image'][:] = 0
        self.assertEqual(int(pair.scene[0, 0, 0]), 21)

    def test_separate_raw_and_live_shape_contracts(self):
        with self.assertRaises(ValueError): CameraPair.capture(observation(128), 0)
        CameraPair.capture(observation(224), 0)
        with self.assertRaises(ValueError): OfflineCameraPair.capture(observation(224), 0)
        OfflineCameraPair.capture(observation(128), 0)

    def test_bad_dtype_size_frame_rejected(self):
        for size in (127, 224, 256):
            with self.assertRaises(ValueError): OfflineCameraPair.capture(observation(size), 0)
        obs = observation(); obs['wrist_image'] = obs['wrist_image'].astype(np.float32)
        with self.assertRaises(ValueError): OfflineCameraPair.capture(obs, 0)
        for frame in (-1, 2, True):
            with self.assertRaises(ValueError): OfflineCameraPair.capture(observation(), frame)

    def test_current_history_prepared_once_for_raw_and_live(self):
        for size, pair_type in ((128, OfflineCameraPair), (224, CameraPair)):
            obs = observation(size); calls = []
            def prepare(images, cfg):
                calls.append([im.shape for im in images])
                return [np.full((224, 224, 3), int(im[0, 0, 0]), np.uint8) for im in images]
            current, past = prepare_query_camera_pairs(obs, pair_type.capture(obs, 0), prepare, object())
            self.assertEqual(calls, [[(size, size, 3)] * 2] * 2)
            self.assertEqual([int(x[0, 0, 0]) for x in (*current, *past)], [21, 73, 21, 73])

    def make_fixture(self, root):
        manifest = dict(data_root_relative='.', inputs=[])
        with h5py.File(root / 'inputs.hdf5', 'w') as f:
            for i in range(8):
                g = f.create_group(f'data/demo_{i}/obs')
                for name in ('agentview_rgb', 'eye_in_hand_rgb'):
                    g.create_dataset(name, data=np.full((2, 128, 128, 3), i + 10, np.uint8))
                g.create_dataset('ee_states', data=np.zeros((2, 6), np.float64))
                g.create_dataset('gripper_states', data=np.zeros((2, 2), np.float64))
                manifest['inputs'].append(dict(trajectory_id=str(i), split='train', source_path='inputs.hdf5',
                    original_trajectory_id=f'demo_{i}', step_count=2, language_instruction='pick up bowl',
                    normalization_statistics_key='libero_spatial_no_noops'))
        return manifest

    def test_reads_both_frames_and_all_input_fields_without_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory); manifest = self.make_fixture(root)
            observations, audit = load_offline_inputs(root, manifest)
            self.assertEqual((len(observations), len(audit)), (8, 16))
            self.assertEqual([a['frame'] for a in audit], [0, 1] * 8)
            def prepare(images, cfg):
                return [np.full((224, 224, 3), int(im[0, 0, 0]), np.uint8) for im in images]
            checks = audit_policy_preprocessing(observations, prepare, SimpleNamespace(center_crop=True))
            self.assertEqual(len(checks), 16)

    def test_reject_split_escape_short_length_and_nonfinite_state(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory); manifest = self.make_fixture(root)
            for key, value in (('split', 'locked'), ('source_path', '../unrelated'), ('step_count', 1)):
                bad = copy.deepcopy(manifest); bad['inputs'][0][key] = value
                with self.assertRaises(ValueError): load_offline_inputs(root, bad)
            with h5py.File(root / 'inputs.hdf5', 'a') as f:
                f['data/demo_0/obs/ee_states'][1, 0] = np.nan
            with self.assertRaises(ValueError): load_offline_inputs(root, manifest)


if __name__ == '__main__': unittest.main()
