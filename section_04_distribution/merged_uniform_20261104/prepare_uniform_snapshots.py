"""Reproduce the paired display particles; no fitting or particle jitter."""
import numpy as np
import compute_uniform as c
if __name__=='__main__':
    rng=np.random.default_rng(20261104)
    labels=rng.integers(c.N,size=512)
    noise=rng.normal(size=(512,2))
    raw_A=c.P[labels]+.35*noise
    raw_B=c.P[labels]+.04*noise
    np.savez(c.ROOT/'snapshot_data.npz',training=c.P,raw_A=raw_A,tweedie_A=c.tweedie(raw_A,.35),raw_B=raw_B,tweedie_B=c.tweedie(raw_B,.04),labels=labels,base_noise=noise)
    print('Saved 512 paired Gaussian-mixture particles and exact empirical-Tweedie images.')
