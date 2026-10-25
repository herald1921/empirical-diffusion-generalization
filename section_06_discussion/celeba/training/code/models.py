import torch
from torch import nn
from velocity_unet import VelocityUNet80,ModelConfig,SinusoidalTimeEmbedding
class VelocityMLP(nn.Module):
    def __init__(self):
        super().__init__();d=6400;e=256
        self.time_embed=SinusoidalTimeEmbedding(e,1000.)
        self.time_skip=nn.Linear(e,1);nn.init.zeros_(self.time_skip.weight);nn.init.zeros_(self.time_skip.bias)
        self.in_linear=nn.Linear(d,d);self.norm=nn.LayerNorm(d)
        self.film=nn.Linear(e,2*d);self.act=nn.SiLU();self.out_linear=nn.Linear(d,d)
        nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.out_linear.weight);nn.init.zeros_(self.out_linear.bias)
    def forward(self,x,t):
        x=x.reshape(len(x),6400);t=torch.as_tensor(t,device=x.device,dtype=torch.float32)
        if t.ndim==0:t=t.expand(len(x))
        emb=self.time_embed(t);scale,shift=self.film(emb).chunk(2,dim=1)
        h=self.norm(self.in_linear(x))*(1+scale)+shift
        # Full-dimensional linear velocity path; no rank-restricted output space.
        return (-1+self.time_skip(emb))*x+self.out_linear(self.act(h))
def make_model(kind):return VelocityUNet80(ModelConfig()) if kind=='unet' else VelocityMLP()

