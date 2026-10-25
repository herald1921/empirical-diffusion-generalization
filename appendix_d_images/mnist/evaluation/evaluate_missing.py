"""Read original MNIST artifacts; write only into a new, isolated output directory.

No training, dataset copying, or calls to the original load_data()/main().
Run with the original CUDA PyTorch environment. --smoke evaluates one time.
"""
import argparse, hashlib, importlib.util, json, time, csv, platform
from pathlib import Path
import numpy as np
import torch


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def nearest_direct(x, reference):
    """Independent direct float64 differences; no squared-norm cancellation."""
    distances, indices = [], []
    for query in x.double():
        dd = (reference.double() - query).square().mean(1)
        val, ix = dd.min(0)
        distances.append(val.sqrt()); indices.append(ix)
    return torch.stack(distances), torch.stack(indices)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/root/dm_jmlr_complete_20260924/mnist')
    ap.add_argument('--raw', default='/root/shared-nvme/experiment-real-data/datasets/mnist/MNIST/raw')
    ap.add_argument('--output', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    source, raw, out = Path(args.source), Path(args.raw), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    target = out / ('smoke.json' if args.smoke else 'evaluation_complete.json')
    if target.exists():
        print('Already completed:', target, flush=True); return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    spec = importlib.util.spec_from_file_location('original_mnist', source/'code/run_mnist.py')
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    protocol = json.loads((source/'protocol.json').read_text())
    assert sha(source/'code/run_mnist.py') == protocol['code_sha256']
    expected = dict(steps=100000,batch_size=128,learning_rate=2e-4,weight_decay=0.,
                    ema_decay=.9999,gradient_clip=1.,model_seed=0,split_seed=20260924,
                    samples=256,sampler_intervals=320,tau_end=.999)
    assert all(protocol[k] == v for k,v in expected.items())
    contract = json.loads((source/'data_contract.json').read_text())
    image_path = raw/'train-images-idx3-ubyte'
    assert sha(image_path) == contract['raw_sha256'][image_path.name]
    images = np.fromfile(image_path, dtype=np.uint8, offset=16).reshape(60000,784)
    part = dict(np.load(source/'data/partitions.npz'))
    perm = np.random.default_rng(20260924).permutation(60000)
    assert np.array_equal(part['s1'],perm[:30000]) and np.array_equal(part['s2'],perm[30000:])
    assert len(np.intersect1d(part['s1'],part['s2'])) == 0
    groups = {g:torch.from_numpy(images[part[g]].copy()).float().div_(255).cuda() for g in ['s1','s2']}
    weights = {}; models = {}
    for kind in ['mlp','unet']:
        folder = source/'runs'/kind/'N10000/s1'
        config = json.loads((folder/'config.json').read_text())
        complete = json.loads((folder/'complete.json').read_text())
        checkpoint = folder/'ema_100000.pt'
        digest = sha(checkpoint)
        assert digest == complete['checkpoint_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['step']==100000 and state['signature']==complete['signature']==config['signature']
        assert all(config['config'][k]==v for k,v in expected.items())
        model = original.model(kind).cuda().eval()
        model.load_state_dict(state['ema'], strict=True)
        assert sum(p.numel() for p in model.parameters()) == config['parameters']
        assert all(p.dtype==torch.float32 for p in model.parameters())
        models[kind] = model
        weights[kind] = dict(path=str(checkpoint),sha256=digest,step=state['step'],weights='ema',
                             parameters=config['parameters'],signature=state['signature'])
    sample = dict(np.load(source/'data/mlp/N10000/s1/samples.npz'))
    noise = torch.from_numpy(sample['noise'][:32].copy()).cuda()
    expected_noise = torch.stack([torch.randn(784,generator=torch.Generator().manual_seed(i)) for i in range(32)])
    assert np.array_equal(noise.cpu().numpy(),expected_noise.numpy())
    # For N=10000 the corresponding reference set is the prefix S2[:10000].
    # Take its final 32 observations, preserving their stored order.
    reference_ids = part['s2'][:10000][-32:]
    reference = groups['s2'][:10000][-32:]
    empirical = original.Empirical(groups['s1'][:10000])
    times = [.3] if args.smoke else [.02,.1,.2,.3,.5,.8,.95]
    fields = {'empirical':empirical, **models}
    arrays = dict(tau=np.array(times),noise=noise.cpu().numpy(),reference=reference.cpu().numpy(),
                  reference_global_indices=reference_ids,seed_ids=np.arange(32),
                  train_global_indices=part['s1'][:10000])
    velocities = {k:[] for k in fields}; noisy=[]; rows=[]; clean_identity_errors=[]
    for t in times:
        x = (1-t)*noise + t*reference; noisy.append(x.cpu().numpy())
        predictions = {k:field(x,t) for k,field in fields.items()}
        for k,v in predictions.items():
            assert v.shape==(32,784) and torch.isfinite(v).all()
            velocities[k].append(v.cpu().numpy())
        for a,b in [('mlp','unet'),('empirical','mlp'),('empirical','unet')]:
            v1,v2=predictions[a].double(),predictions[b].double()
            value=float((1-t)*(v1-v2).square().mean().sqrt())
            clean1=x.double()+(1-t)*v1;clean2=x.double()+(1-t)*v2
            identity_error=abs(value-float((clean1-clean2).square().mean().sqrt()))
            assert identity_error<1e-12
            clean_identity_errors.append(identity_error)
            rows.append(dict(tau=t,pair=a+'_'+b,scaled_field_rmse=value,n_inputs=32,dimension=784))
        print('COMMON_INPUT',t,rows[-3:],flush=True)
    if args.smoke:
        dump(target,dict(passed=True,seconds=time.time()-started,weights=weights,
                         finite_fields=True,scaled_velocity_equals_clean_rmse=True))
        print('SMOKE PASSED',time.time()-started,flush=True);return
    arrays['noisy_inputs']=np.stack(noisy)
    for k,v in velocities.items():arrays['velocity_'+k]=np.stack(v)
    np.savez_compressed(out/'field_inputs_predictions.npz',**arrays)
    with (out/'field_discrepancies.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(out/'field_metadata.json',dict(weights=weights,common_input_formula='x_tau=(1-tau)*G+tau*Z',
        reference_selection='last 32 images of the prescribed N=10000 reference prefix S2[:10000]',
        reference_global_indices=reference_ids.tolist(),seeds=list(range(32)),tau=times,
        metric='(1-tau)*sqrt(mean over all 32 inputs and 784 pixels of (v_A-v_B)^2)',
        inference='FP32 neural inference, TF32 disabled; empirical posterior likelihoods and mean accumulated in float64, returned as FP32, as original implementation',
        confidence_intervals='none, as Appendix C.6',source_code=str(source/'code/run_mnist.py'),
        source_sha256=sha(source/'code/run_mnist.py'),dataset_sha256=sha(image_path),
        torch_version=torch.__version__,device=torch.cuda.get_device_name(0)))
    # Validate every reused configuration; independently test actual posterior weights
    # on every saved raw path (8/direction), all times. All 256 ESS paths are kept.
    checks=[]; noise_all=sample['noise'];tau_all=sample['tau']
    for kind in ['empirical','mlp','unet']:
        for n in [100,1000,10000]:
            for g in ['s1','s2']:
                folder=source/'data'/kind/f'N{n}'/g
                complete=json.loads((folder/'complete.json').read_text())
                z=dict(np.load(folder/'samples.npz'))
                assert sha(folder/'samples.npz')==complete['samples_sha256']
                assert np.array_equal(z['noise'],noise_all) and np.array_equal(z['tau'],tau_all)
                ref=groups[g][:n];other=groups['s2' if g=='s1' else 's1'][:n]
                assert len(ref)==len(other)==n
                check=dict(method=kind,n=n,group=g,samples_sha256=complete['samples_sha256'])
                if kind!='empirical':
                    cp=source/'runs'/kind/f'N{n}'/g
                    config=json.loads((cp/'config.json').read_text());finish=json.loads((cp/'complete.json').read_text())
                    checkpoint=cp/'ema_100000.pt'
                    assert all(config['config'][k]==v for k,v in expected.items())
                    assert finish['step']==100000 and finish['checkpoint_sha256']==complete['checkpoint_sha256']
                    check['checkpoint_path']=str(checkpoint);check['checkpoint_present']=checkpoint.exists()
                    if checkpoint.exists():assert sha(checkpoint)==complete['checkpoint_sha256']
                    check['checkpoint_sha256']=complete['checkpoint_sha256']
                else:
                    e=original.Empirical(ref);ess_errors=[];mean_errors=[];norm_errors=[]
                    for i,t in enumerate(tau_all):
                        x=torch.from_numpy(z['raw_trace'][i]).cuda()
                        # Independent distance-based likelihood calculation; centering
                        # avoids underflow. Unlike production, includes ||x||^2 term.
                        q=x.double();r=ref.double()
                        dist=(q[:,None,:]-float(t)*r[None,:,:]).square().sum(2)
                        logw=-dist/(2*(1-float(t))**2)
                        logw=logw-logw.max(1,keepdim=True).values
                        w=logw.exp();w=w/w.sum(1,keepdim=True)
                        ess=1/w.square().sum(1);mean=w@r
                        norm_errors.append(float(abs(w.sum(1)-1).max()))
                        expected_ess=torch.from_numpy(z['ess'][i,:8]).cuda().double()
                        ess_errors.append(float((abs(ess-expected_ess)/expected_ess).max()))
                        expected_mean=torch.from_numpy(z['readout'][i,:8]).cuda().double()
                        mean_errors.append(float(abs(mean-expected_mean).max()))
                    assert max(ess_errors)<2e-6 and max(mean_errors)<2e-6 and max(norm_errors)<1e-12
                    check.update(ess_max_relative_error=max(ess_errors),posterior_readout_max_abs_error=max(mean_errors),
                        weights_max_normalization_error=max(norm_errors),ess_independent_paths=8,ess_saved_paths=256,
                        ess_times_checked=len(tau_all),all_ess_bounds=bool(np.all(z['ess']>=1-1e-6) and np.all(z['ess']<=n+1e-3)))
                # Direct full-set nearest search for the four first seeds at selected
                # readout and terminal times in each direction, including every gallery.
                errors=[]
                for i in sorted(set([9,19])):
                    x=torch.from_numpy(z['readout'][i,:4]).cuda()
                    for metric,refs,key in [('dtrain',ref,'nearest_train_indices'),('dref',other,'nearest_reference_indices')]:
                        d,ix=nearest_direct(x,refs)
                        err=float(np.max(abs(d.cpu().numpy()-z[metric][i,:4])))
                        assert err<2e-7
                        # Duplicate pixels permit tied nearest indices; verify distance.
                        saved=refs[torch.from_numpy(z[key][i,:4]).cuda()]
                        ds=(x.double()-saved.double()).square().mean(1).sqrt()
                        assert float(abs(ds-d).max())<1e-12
                        errors.append(err)
                check['direct_float64_nn_max_abs_error']=max(errors)
                checks.append(check);dump(out/'verification_progress.json',checks)
                print('VERIFIED',kind,n,g,flush=True)
    dump(out/'remote_validation.json',dict(passed=True,split_reproduced=True,disjoint_index_groups=True,
        no_labels_augmentation_or_pca_in_training=True,paired_cpu_seed_inputs_exact=True,
        source_code_sha256=sha(source/'code/run_mnist.py'),checks=checks,
        common_input_clean_identity_max_error=max(clean_identity_errors)))
    dump(target,dict(passed=True,seconds=time.time()-started,source=str(source),output=str(out),
        training_runs_started=0,new_evaluation='32 common inputs, 7 linear FM times, 3 fields',
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['evaluation.log','evaluation_complete.json']}))
    print('EVALUATION COMPLETE',time.time()-started,flush=True)


if __name__=='__main__':main()
