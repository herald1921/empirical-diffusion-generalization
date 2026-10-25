"""Freeze exact numerical inputs for the composite, without editing source artifacts."""
from pathlib import Path
import argparse, csv, hashlib, json, shutil
import numpy as np

OUT=Path(__file__).resolve().parent
DEFAULT=OUT.parents[1]/'review_rounds/20261008_usb_data_inventory/library/01_current_research/DM_JMLR_research_bundle/experiments/mnist'
NS=[100,1000,10000];METHODS=['empirical','mlp','unet'];GROUPS=['s1','s2']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,o):Path(p).write_text(json.dumps(o,indent=2)+'\n',encoding='utf-8')

def paired_statistics(tr,ref,ids):
    # Average directions inside each seed, then resample those 256 paired units.
    tr=tr.mean(axis=0,dtype=np.float64);ref=ref.mean(axis=0,dtype=np.float64)
    allv=np.stack([tr,ref,ref-tr])
    means=allv.mean(-1)
    bands=np.stack([np.quantile(x[:,ids].mean(-1),[.025,.975],axis=1) for x in allv])
    return means,bands

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=DEFAULT)
    args=ap.parse_args();source=args.source.resolve()
    assert source.is_dir()
    protocol=read(source/'protocol.json');assert protocol['code_sha256']==sha(source/'code/run_mnist.py')
    part=dict(np.load(source/'data/partitions.npz'))
    permutation=np.random.default_rng(20260924).permutation(60000)
    assert np.array_equal(part['s1'],permutation[:30000]) and np.array_equal(part['s2'],permutation[30000:])
    assert np.intersect1d(part['s1'],part['s2']).size==0
    manifest=[];cases={};training=[];noise=None;tau=None;checks=[]
    for kind in METHODS:
        for n in NS:
            for g in GROUPS:
                folder=source/'data'/kind/f'N{n}'/g;p=folder/'samples.npz'
                complete=read(folder/'complete.json');digest=sha(p)
                assert digest==complete['samples_sha256']
                z=dict(np.load(p));assert z['readout'].shape==(20,256,784)
                assert z['dtrain'].shape==z['dref'].shape==(20,256)
                assert all(np.isfinite(v).all() for v in z.values())
                if noise is None:noise=z['noise'];tau=z['tau']
                assert np.array_equal(noise,z['noise']) and np.array_equal(tau,z['tau'])
                assert np.allclose(z['reverse_t'],2/np.pi*np.arctan(tau/(1-tau)),atol=1e-14)
                neighbor_error=[]
                for key,distance in [('train','dtrain'),('reference','dref')]:
                    ix=z['nearest_'+key+'_indices'];assert ix.min()>=0 and ix.max()<n
                    error=np.abs(np.sqrt(np.mean((z['readout'][:,:8].astype('float64')-
                          z['display_'+key+'_neighbors'].astype('float64'))**2,axis=2))-z[distance][:,:8]).max()
                    assert error<2e-7;neighbor_error.append(float(error))
                if kind=='empirical':
                    assert z['ess'].shape==(20,256) and np.allclose(z['ess'][0],n)
                    assert np.all(z['ess']>=1-1e-6) and np.all(z['ess']<=n+1e-3)
                else:
                    run=source/'runs'/kind/f'N{n}'/g
                    config=read(run/'config.json');finish=read(run/'complete.json')
                    assert finish['step']==100000 and finish['checkpoint_sha256']==complete['checkpoint_sha256']
                    assert config['kind']==kind and config['n']==n and config['group']==g
                    for k,v in dict(steps=100000,batch_size=128,learning_rate=2e-4,weight_decay=0.,
                        ema_decay=.9999,gradient_clip=1.,model_seed=0,split_seed=20260924,
                        samples=256,sampler_intervals=320,tau_end=.999).items():assert config['config'][k]==v
                    training.append(dict(method=kind,n=n,group=g,config=config,complete=finish,
                        cloud_checkpoint=f'/root/dm_jmlr_complete_20260924/mnist/runs/{kind}/N{n}/{g}/ema_100000.pt'))
                manifest.append(dict(method=kind,n=n,group=g,path=str(p),sha256=digest,
                    bytes=p.stat().st_size,checkpoint_sha256=complete['checkpoint_sha256']))
                checks.append(dict(method=kind,n=n,group=g,neighbor_max_abs_error=max(neighbor_error),
                    unclipped_range=[float(z['readout'].min()),float(z['readout'].max())],
                    solver_refinement=read(folder/'solver_refinement.json')))
                cases[kind,n,g]=z
    bootstrap_ids=np.random.default_rng(22092027).integers(0,256,(2000,256))
    arrays=dict(tau=tau,seed_ids=np.arange(256),noise=noise,bootstrap_indices=bootstrap_ids,
        s1_indices=part['s1'],s2_indices=part['s2'],sample_sizes=np.array(NS),
        snapshot_tau=np.array([.05,.25,.5,.999]),method_names=np.array(METHODS),
        metric_names=np.array(['dtrain','dref','gap']))
    rows=[];ess_rows=[];a=[];c=[];e=[]
    for n in NS:
        z=cases['empirical',n,'s1']
        take=[int(np.flatnonzero(tau==t)[0]) for t in arrays['snapshot_tau']]
        a.append(z['readout'][take,0])
        ess=np.concatenate([cases['empirical',n,g]['ess'] for g in GROUPS],axis=1)
        arrays[f'ess_N{n}']=ess
        arrays[f'ess_quantiles_N{n}']=np.quantile(ess,[.25,.5,.75],axis=1)
        for ti,t in enumerate(tau):
            q=arrays[f'ess_quantiles_N{n}'][:,ti]
            ess_rows.append(dict(n=n,tau=float(t),q25=q[0],median=q[1],q75=q[2],paths=512))
        cc=[]
        for kind in METHODS:
            s1,s2=[cases[kind,n,g] for g in GROUPS]
            tr=np.stack([s1['dtrain'],s2['dtrain']]);ref=np.stack([s1['dref'],s2['dref']])
            arrays[f'{kind}_N{n}_dtrain']=tr;arrays[f'{kind}_N{n}_dref']=ref
            means,bands=paired_statistics(tr,ref,bootstrap_ids)
            arrays[f'{kind}_N{n}_means']=means;arrays[f'{kind}_N{n}_ci95']=bands
            # Original code computes float32 direction means. The new float64
            # aggregate differs only by rounding and preserves identical units.
            oldgap=np.mean([s1['dref'],s2['dref']],axis=0)-np.mean([s1['dtrain'],s2['dtrain']],axis=0)
            assert np.max(abs(oldgap.mean(1)-means[2]))<1e-7
            for ti,t in enumerate(tau):
                for m,key in enumerate(['dtrain','dref','gap']):
                    rows.append(dict(method=kind,n=n,tau=float(t),metric=key,mean=means[m,ti],
                        lower95=bands[m,0,ti],upper95=bands[m,1,ti],paired_seeds=256,directions=2))
            k=int(np.flatnonzero(tau==.25)[0]) if kind=='empirical' else len(tau)-1
            cc.append(np.stack([s1['display_train_neighbors'][k,0],s1['readout'][k,0],
                s2['readout'][k,0],s2['display_train_neighbors'][k,0]]))
        c.append(cc)
    for kind in ['mlp','unet']:e.append(cases[kind,10000,'s1']['readout'][-1,:4])
    arrays.update(panel_a=np.array(a),panel_c=np.array(c),panel_e=np.array(e),panel_e_seeds=np.arange(4))
    np.savez_compressed(OUT/'plot_data.npz',**arrays)
    for name,records in [('distance_summary.csv',rows),('ess_summary.csv',ess_rows)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    provenance=OUT/'provenance';provenance.mkdir(exist_ok=True)
    for name in ['protocol.json','data_contract.json','data_duplicate_audit.json']:
        shutil.copyfile(source/name,provenance/name)
    for name in ['run_mnist.py','plot_mnist.py','audit_mnist.py']:
        shutil.copyfile(source/'code'/name,provenance/('original_'+name))
    dump(OUT/'source_manifest.json',manifest)
    dump(OUT/'training_metadata.json',training)
    dump(OUT/'local_validation.json',dict(passed=True,configurations=18,neural_runs=12,
        split_seed=20260924,permutation_exact=True,index_disjoint=True,paired_noise_exact=True,
        source_code_hash_verified=True,source_sample_hashes_verified=True,checks=checks))
    dump(OUT/'metadata.json',dict(title='MNIST: empirical concentration and learned generation',
        manuscript=r'local-archive/Topology_and_Distributional_Generalization_in_Empirical_Diffusion_Models.pdf',
        manuscript_pages=[41,42,43,45,46],source=str(source),sample_sizes=NS,
        time_coordinate='linear FM tau; do not use archived angular reverse_t',
        empirical_readout_tau=.25,neural_readout_tau=.999,snapshot_tau=[.05,.25,.5,.999],
        images=dict(panel_a='S1, seed 0 for every N and time',panel_c='seed 0 shared by S1/S2 and all methods',
            panel_c_order=['nearest S1 to S1 output','S1 output','paired S2 output','nearest S2 to its own output'],
            panel_e='S1, N=10000, seeds 0,1,2,3 selected before inspection',display='clip [0,1] only, fixed grayscale'),
        distance='min over N candidates of sqrt(mean_784((unclipped_clean_readout-reference_image)^2))',
        gap='dref-dtrain; average directions equally within each seed before averaging seeds',
        bootstrap=dict(seed=22092027,replicates=2000,unit='paired seed (retains both directions, methods and times)',
            bands='pointwise percentile 95%, conditional on these fixed models and datasets',
            note='Original Figure 9 used bands for gaps only; this composite also applies the same bootstrap to both distances.'),
        ess='1/sum(w**2), normalized empirical posterior weights at raw states; median/IQR over 512 paths',
        training_runs_started=0,source_precision='Original saved measurements retained; float64 aggregation and bootstrap',
        archived_time_mapping='reverse_t=(2/pi)*atan(tau/(1-tau)); only tau plotted'))
    print('Prepared exact plot data:',OUT/'plot_data.npz',flush=True)

if __name__=='__main__':main()
