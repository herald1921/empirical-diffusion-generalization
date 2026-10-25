from pathlib import Path
import sys, unittest, zipfile, tempfile, importlib.util, hashlib
import numpy as np
import torch
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'section_06_discussion/celeba/metrics/code'))
sys.path.insert(0,str(R/'shared/celeba_preprocessing'))
from nearest_exact import nearest_rmse
from paper_contract import load_contract, ordered_sources, validate_shard
from prepare_celeba80_paper import save_partitions, EXPECTED_CLEAN_COUNT, PAPER_HALF_SIZE
torch.set_num_threads(2)

class NearestTests(unittest.TestCase):
    def test_close_candidates_regression(self):
        torch.manual_seed(123)
        q=torch.rand(1,6400);ref=q.repeat(64,1)+1e-5*torch.randn(64,6400);ref[0]=q[0]
        value,ix=nearest_rmse(q,ref)
        self.assertEqual(int(ix[0]),0);self.assertEqual(float(value[0]),0.)

    def test_chunk_invariance_and_independent_reference(self):
        torch.manual_seed(17);q=torch.randn(7,31);ref=torch.randn(43,31)
        expected=(q.double()[:,None]-ref.double()).square().mean(-1).sqrt()
        for chunk in [1,6,43,100]:
            value,ix=nearest_rmse(q,ref,chunk=chunk,query_chunk=3)
            torch.testing.assert_close(value,expected.min(1).values,rtol=1e-14,atol=1e-14)
            self.assertTrue(torch.equal(ix,expected.argmin(1)))

    def test_ties_first_global_index(self):
        q=torch.zeros(2,3);ref=torch.ones(8,3)
        ref[3]=0.;ref[7]=0.
        value,ix=nearest_rmse(q,ref,chunk=2)
        self.assertTrue(torch.equal(ix,torch.tensor([3,3])))
        self.assertEqual(float(value.max()),0.)

    def test_float64_subtraction_before_reduction(self):
        q=torch.tensor([[1e8+1.,1e8+2.]],dtype=torch.float64)
        ref=torch.tensor([[1e8+1.,1e8+3.],[1e8+1.,1e8+2.]],dtype=torch.float64)
        value,ix=nearest_rmse(q,ref,chunk=1)
        self.assertEqual(int(ix[0]),1);self.assertEqual(float(value[0]),0.)

    def test_empty_query(self):
        d,i=nearest_rmse(torch.empty(0,3),torch.zeros(2,3))
        self.assertEqual(len(d),0);self.assertEqual(len(i),0)

    def test_invalid_inputs(self):
        for q,r in [(torch.zeros(1,3),torch.empty(0,3)),(torch.zeros(1,3),torch.zeros(1,2)),(torch.tensor([[float('nan')]]),torch.ones(1,1))]:
            with self.assertRaises(ValueError):nearest_rmse(q,r)

class DataContractTests(unittest.TestCase):
    def test_released_order_and_partition_hashes(self):
        c,names,removed=load_contract()
        self.assertEqual(len(names),202599);self.assertEqual(len(removed),6552)
        self.assertEqual(c['retained_count'],EXPECTED_CLEAN_COUNT)
        self.assertEqual(c['group_size'],PAPER_HALF_SIZE)

    def test_new_partition_metadata_and_exact_indices(self):
        c,_,removed=load_contract();mask=np.zeros(c['raw_count'],bool);mask[removed]=True
        # Retain the small generated index files for inspection; no cleanup/deletion.
        tmp=Path(tempfile.mkdtemp(prefix='paper-partition-check-'))
        info=save_partitions(tmp,mask,EXPECTED_CLEAN_COUNT)
        self.assertEqual(info['paper_half_size'],98023)
        with np.load(tmp/'paper_partitions.npz') as p:
            self.assertEqual(p['paper_subset_sizes'].tolist(),[1,10,100,1000,10000,98023])
            for key,h in c['partition_array_sha256'].items():self.assertEqual(hashlib.sha256(p[key].tobytes()).hexdigest(),h)
            self.assertEqual(np.intersect1d(p['group_a_raw_indices'],p['group_b_raw_indices']).size,0)

    def test_order_is_contract_not_archive_sequence(self):
        entries=[zipfile.ZipInfo(x) for x in ['images/b.jpg','images/a.jpg']]
        self.assertEqual([x.filename for x in ordered_sources(entries,['a.jpg','b.jpg'])],['images/a.jpg','images/b.jpg'])
        with self.assertRaises(ValueError):ordered_sources(entries,['b.jpg','missing.jpg'])

    def test_changed_pixel_content_rejected(self):
        a=np.zeros((1,80,80),np.uint16)
        record={'filename':'test.npz','count':1,'array_sha256':hashlib.sha256(a.tobytes()).hexdigest()}
        validate_shard(a,record);a[0,0,0]=1
        with self.assertRaises(ValueError):validate_shard(a,record)

if __name__=='__main__':unittest.main(verbosity=2)
