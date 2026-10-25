import numpy as np,torch

def schedule(points=160,end=.999,captures=()):
 return np.array(sorted(set(np.linspace(0,end,points+1).tolist()+list(captures))))

@torch.inference_mode()
def integrate(field,noise,grid,captures=()):
 x=noise.clone();saved={}
 for t0,t1 in zip(grid[:-1],grid[1:]):
  v=field(x,float(t0));pred=x+(t1-t0)*v
  x=x+.5*(t1-t0)*(v+field(pred,float(t1)))
  if t1 in captures:saved[float(t1)]=(x.clone(),x+(1-t1)*field(x,float(t1)))
 return x,x+(1-grid[-1])*field(x,float(grid[-1])),saved

def empirical_field(ref):
 norms=torch.cat([r.double().square().sum(1).float() for r in ref.split(4096)])
 def field(x,t):
  assert 0<=t<1
  logits=(2*t*(x@ref.T)-t*t*norms[None,:])/(2*(1-t)**2)
  mean=logits.softmax(1)@ref
  return (mean-x)/(1-t)
 return field
