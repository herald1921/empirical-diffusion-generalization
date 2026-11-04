"""Readouts needed for ambient-dimension comparison; no new geometry."""
import compute_uniform as c
import numpy as np
if __name__=='__main__':
    c.run([(.16,14,8192,c.SEED,'late_stop_refined'),(.16,12,4096,c.SEED+1,'late_stop_scramble')],'late_stop_checks.csv',2)
    grid=np.r_[np.arange(.02,.091,.0025),[c.R*np.sin(np.pi/c.N)/(np.sqrt(d)+np.sqrt(2*np.log(20))) for d in [2,4,8,16]]]
    c.run([(float(e),12,4096,c.SEED,'dimension_crossings') for e in grid],'dimension_checks.csv',3)
