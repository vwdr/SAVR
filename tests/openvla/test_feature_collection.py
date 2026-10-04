import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import collect_current_frame_features as worker


class CollectionTests(unittest.TestCase):
    def valid(self):
        samples=[dict(sample_id=str(i),learning_role='fit' if i%100<80 else 'validation') for i in range(4000)]
        m=dict(samples=samples)
        rows=[dict(r,artifact='features/'+str(i).zfill(4)+'.npz',model_queries=2*(i+1),
                   input_unchanged=True,record_roundtrip_equal=True,query_counts_equal_one=True,
                   no_backbone_gradients=True) for i,r in enumerate(samples)]
        s=dict(complete=True,model_queries=8000,records=4000,optimizer_steps=0,episodes=0,
               positive_method_result=False,automatic_retry=False,authenticated_files_unchanged=True,
               elapsed_seconds=9000,peak_aggregate_gpu_memory_mib=16000,artifact_bytes_before_summary=17100000000)
        return {},m,s,rows

    def test_complete_contract(self):self.assertTrue(worker.reconcile(*self.valid()))

    def test_role_corruption_or_omission_rejected(self):
        c,m,s,r=self.valid();r[0]['learning_role']='validation'
        with self.assertRaises(ValueError):worker.reconcile(c,m,s,r)
        c,m,s,r=self.valid()
        with self.assertRaises(ValueError):worker.reconcile(c,m,s,r[:-1])

    def test_gradient_input_or_accounting_failure_rejected(self):
        for key in ('input_unchanged','record_roundtrip_equal','no_backbone_gradients'):
            c,m,s,r=self.valid();r[0][key]=False
            with self.assertRaises(ValueError):worker.reconcile(c,m,s,r)
        c,m,s,r=self.valid();s['model_queries']=7999
        with self.assertRaises(ValueError):worker.reconcile(c,m,s,r)

    def test_caps_and_invalid_numbers_rejected(self):
        for key,value in (('elapsed_seconds',28800),('peak_aggregate_gpu_memory_mib',23552),
                          ('artifact_bytes_before_summary',20*1024**3),('elapsed_seconds',float('nan'))):
            c,m,s,r=self.valid();s[key]=value
            with self.assertRaises(ValueError):worker.reconcile(c,m,s,r)


if __name__=='__main__':unittest.main()
