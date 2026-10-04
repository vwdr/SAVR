import unittest
from savr.openvla.spatial_selection import stratified_visual_positions

class SelectionTests(unittest.TestCase):
    def test_counts_order_camera_balance_and_strata(self):
        for budget in (256,384,512):
            ids=stratified_visual_positions(budget)
            self.assertEqual(len(ids),budget)
            self.assertEqual(ids,tuple(sorted(set(ids))))
            self.assertEqual(sum(i<=256 for i in ids),budget//2)
            for camera in range(2):
                for row in range(0,16,2):
                    for col in range(0,16,2):
                        cell={1+256*camera+16*(row+dr)+col+dc for dr in (0,1) for dc in (0,1)}
                        self.assertEqual(len(cell.intersection(ids)),budget//128)
    def test_dense_identity_and_determinism(self):
        self.assertEqual(stratified_visual_positions(512),tuple(range(1,513)))
        self.assertEqual(stratified_visual_positions(256),stratified_visual_positions(256))
    def test_invalid_budgets(self):
        for budget in (True,256.,0,128,768):
            with self.assertRaises(ValueError):stratified_visual_positions(budget)

if __name__=='__main__':unittest.main()
