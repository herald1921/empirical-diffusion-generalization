"""Small numerical and provenance tests. Toy values are never used in the figure."""
from pathlib import Path
import csv,json,unittest
import numpy as np
from prepare_data import paired_statistics,sha

ROOT=Path(__file__).resolve().parent

def posterior(x,ref,t):
    logw=-np.sum((x[:,None,:]-t*ref[None,:,:])**2,axis=2)/(2*(1-t)**2)
    w=np.exp(logw-logw.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
    return w,w@ref,1/np.sum(w*w,axis=1)

class Numerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.z=dict(np.load(ROOT/'frozen_data/plot_data.npz'))
        cls.f=dict(np.load(ROOT/'field_inputs_predictions.npz'))

    def test_split_exact_disjoint_nested(self):
        perm=np.random.default_rng(20260924).permutation(60000)
        np.testing.assert_array_equal(self.z['s1_indices'],perm[:30000])
        np.testing.assert_array_equal(self.z['s2_indices'],perm[30000:])
        self.assertEqual(len(np.intersect1d(self.z['s1_indices'],self.z['s2_indices'])),0)
        for n in [100,1000,10000]:
            for g in ['s1','s2']:self.assertEqual(len(np.unique(self.z[g+'_indices'][:n])),n)

    def test_ess_uniform_single_atom_and_normalization(self):
        x=np.array([[0.,1.],[1.,0.]])
        ref=np.array([[-1.,0.],[0.,0.],[1.,0.]])
        w,mean,ess=posterior(x,ref,0.)
        np.testing.assert_allclose(w.sum(1),1,atol=1e-15)
        np.testing.assert_allclose(ess,3)
        _,mean,ess=posterior(x,ref[:1],.9)
        np.testing.assert_allclose(ess,1)
        np.testing.assert_allclose(mean,np.broadcast_to(ref[0],x.shape))

    def test_distance_metric_and_gap_sign(self):
        x=np.array([1.,2.]);train=np.array([[1.,2.],[10.,10.]])
        ref=np.array([[3.,4.],[-10.,-10.]])
        dtrain=np.sqrt(np.min(np.mean((train-x)**2,1)))
        dref=np.sqrt(np.min(np.mean((ref-x)**2,1)))
        self.assertEqual(dtrain,0.);self.assertEqual(dref,2.)
        self.assertEqual(dref-dtrain,2.)
        # Confirms unclipped distance is different from the display-only clipping.
        self.assertNotEqual(np.sqrt(np.mean((x-ref[0])**2)),
                            np.sqrt(np.mean((np.clip(x,0,1)-ref[0])**2)))

    def test_bootstrap_retains_both_directions(self):
        # Opposite directional effects cancel within each paired seed.
        tr=np.zeros((2,1,4));ref=np.array([[[1,2,3,4]],[[-1,-2,-3,-4]]],float)
        ids=np.random.default_rng(12).integers(0,4,(100,4))
        mean,ci=paired_statistics(tr,ref,ids)
        np.testing.assert_array_equal(mean[2],0)
        np.testing.assert_array_equal(ci[2],0)

    def test_common_inputs_and_reference_indices(self):
        np.testing.assert_array_equal(self.f['seed_ids'],np.arange(32))
        np.testing.assert_array_equal(self.f['noise'],self.z['noise'][:32])
        np.testing.assert_array_equal(self.f['reference_global_indices'],self.z['s2_indices'][:10000][-32:])
        for i,t in enumerate(self.f['tau']):
            # Reproduce the FP32 scalar conversions used by PyTorch. NumPy 2
            # otherwise promotes an np.float64 time scalar to a double result.
            x=np.float32(1-t)*self.f['noise']+np.float32(t)*self.f['reference']
            np.testing.assert_array_equal(self.f['noisy_inputs'][i],x)

    def test_pooled_scaled_field_rmse(self):
        records=list(csv.DictReader((ROOT/'frozen_data/field_discrepancies.csv').open()))
        for row in records:
            a,b=row['pair'].split('_');t=float(row['tau'])
            i=int(np.flatnonzero(self.f['tau']==t)[0])
            v1=self.f['velocity_'+a][i].astype('float64')
            v2=self.f['velocity_'+b][i].astype('float64')
            expected=(1-t)*np.sqrt(np.mean((v1-v2)**2))
            self.assertAlmostEqual(expected,float(row['scaled_field_rmse']),places=13)

    def test_plotted_statistics_trace_to_seed_arrays(self):
        for method in ['empirical','mlp','unet']:
            for n in [100,1000,10000]:
                key=f'{method}_N{n}'
                means,bands=paired_statistics(self.z[key+'_dtrain'],self.z[key+'_dref'],self.z['bootstrap_indices'])
                np.testing.assert_array_equal(self.z[key+'_means'],means)
                np.testing.assert_array_equal(self.z[key+'_ci95'],bands)
        for n in [100,1000,10000]:
            np.testing.assert_array_equal(self.z[f'ess_quantiles_N{n}'],
                np.quantile(self.z[f'ess_N{n}'],[.25,.5,.75],axis=1))

    @unittest.skipUnless(all(Path(r['path']).is_file() for r in json.loads((ROOT/'source_manifest.json').read_text())), 'Original large source artifacts are not included; use repository manifest verification for distributed files.')
    def test_source_hashes_and_remote_verification(self):
        for row in json.loads((ROOT/'source_manifest.json').read_text()):
            self.assertEqual(sha(row['path']),row['sha256'])
        remote=json.loads((ROOT/'frozen_data/remote_validation.json').read_text())
        self.assertTrue(remote['passed']);self.assertEqual(len(remote['checks']),18)
        for row in remote['checks']:
            if row['method']!='empirical':self.assertTrue(row['checkpoint_present'])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Numerics)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT/'unit_test_results.json').write_text(json.dumps(dict(tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
    raise SystemExit(0 if result.wasSuccessful() else 1)
