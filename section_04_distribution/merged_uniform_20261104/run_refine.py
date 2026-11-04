import compute_uniform as c, numpy as np
if __name__=='__main__':
 c.run([(float(e),14,8192,c.SEED,'fine_minimum') for e in np.arange(.108,.1141,.0005)],'minimum_search.csv',3)
 c.run([(e,p,n,s,'fine_check') for e in [.11,.35,.5] for p,n,s in [(14,8192,c.SEED),(12,4096,c.SEED+1)]],'fine_checks.csv',3)
