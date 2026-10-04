import unittest
from savr.openvla.current_frame_sampling import select_samples


class SamplingTests(unittest.TestCase):
    def rows(self):
        return [dict(split='train', suite=str(t//10), task_id=str(t), trajectory_id=f'{t}-{d}',
                     split_hash=f'{d:02}', step_count=80, eligible_query_count=10)
                for t in range(40) for d in range(35)]

    def test_deterministic_balanced_and_disjoint(self):
        rows = self.rows(); selected = select_samples(rows)
        self.assertEqual(selected, select_samples(rows[::-1]))
        self.assertEqual(len(selected), 4000)
        fit = {r['identity']['trajectory_id'] for r in selected if r['learning_role']=='fit'}
        val = {r['identity']['trajectory_id'] for r in selected if r['learning_role']=='validation'}
        self.assertFalse(fit & val)
        for t in range(40):
            task = [r for r in selected if r['identity']['task_id']==str(t)]
            self.assertEqual(sum(r['learning_role']=='fit' for r in task), 80)
            self.assertEqual(sum(r['learning_role']=='validation' for r in task), 20)

    def test_locked_rows_not_selected(self):
        rows = self.rows(); locked = dict(rows[0], split='locked_test', trajectory_id='locked')
        self.assertEqual(select_samples(rows), select_samples([locked, *rows]))

    def test_malformed_or_insufficient_rejected(self):
        with self.assertRaises(ValueError): select_samples(self.rows()[:-1])
        rows = self.rows(); rows[0]['step_count'] = 79
        with self.assertRaises(ValueError): select_samples(rows)


if __name__ == '__main__': unittest.main()
