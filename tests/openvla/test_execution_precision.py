import unittest
import numpy as np
from savr.openvla.execution_precision import execution_chunk_float32, processed_command_chunk


class PrecisionTests(unittest.TestCase):
    def test_float64_rounding_matches_reference_bytes(self):
        raw=np.linspace(-.987654321,.987654321,56,dtype=np.float64).reshape(8,7)
        original=raw.copy()
        result=execution_chunk_float32(raw)
        self.assertEqual(result.dtype,np.float32)
        self.assertEqual(result.tobytes(),np.asarray(raw,dtype=np.float32).tobytes())
        self.assertFalse(np.array_equal(raw,result.astype(np.float64)))
        np.testing.assert_array_equal(raw,original)
        self.assertFalse(np.shares_memory(raw,result))

    def test_float32_input_is_copied_without_change(self):
        raw=np.arange(56,dtype=np.float32).reshape(8,7)/100
        result=execution_chunk_float32(raw)
        self.assertEqual(result.tobytes(),raw.tobytes())
        self.assertFalse(np.shares_memory(raw,result))

    def test_rounding_precedes_mutating_gripper_processing(self):
        raw=np.linspace(-.9,.9,56,dtype=np.float64).reshape(8,7)
        raw[:,6]=[.4999999999,.5000000001,.1,.9,.2,.8,.3,.7]
        original=raw.copy(); seen=[]
        def process(a,family):
            seen.append(a.copy())
            self.assertEqual(a.dtype,np.float32)
            a[-1]=-np.sign(2*a[-1]-1)
            return a
        commands=processed_command_chunk(raw,process,"openvla")
        expected=np.asarray(raw,dtype=np.float32).copy()
        expected[:,-1]=-np.sign(2*expected[:,-1]-1)
        self.assertEqual(commands.tobytes(),expected.tobytes())
        self.assertEqual(len(seen),8)
        np.testing.assert_array_equal(raw,original)

    def test_reject_bad_shape_nonfinite_and_overflow(self):
        for raw in (np.zeros((7,7)),np.zeros((8,7),dtype=int),
                    np.full((8,7),np.nan),np.full((8,7),1e300)):
            with self.assertRaises(ValueError): execution_chunk_float32(raw)


if __name__=="__main__": unittest.main()
