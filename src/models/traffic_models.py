"""PyTorch traffic forecasting competitors sharing one tensor contract.

The graph baselines are intentionally dependency-light implementations. They
use the same `(batch, time, features, nodes)` tensor contract so experiments
can compare architecture changes without changing the data path.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

class LSTMOnly(nn.Module):
    def __init__(self,features:int,nodes:int,horizon:int,hidden:int=64,layers:int=2):
        super().__init__(); self.nodes=nodes; self.horizon=horizon; self.lstm=nn.LSTM(features*nodes,hidden,layers,batch_first=True,dropout=0.1 if layers>1 else 0.0); self.head=nn.Linear(hidden,horizon*nodes)
    def forward(self,x):
        batch=x.shape[0]; sequence=x.permute(0,1,3,2).reshape(batch,x.shape[1],-1); state,_=self.lstm(sequence); return self.head(state[:,-1]).reshape(batch,self.horizon,self.nodes)

class AdaptiveGraphTemporal(nn.Module):
    """Graph-temporal forecaster with learned adjacency and node-wise GRU."""
    def __init__(self,features:int,nodes:int,horizon:int,hidden:int=64):
        super().__init__(); self.nodes=nodes; self.horizon=horizon; self.projection=nn.Linear(features,hidden); self.adaptive_logits=nn.Parameter(torch.zeros(nodes,nodes)); self.gru=nn.GRU(hidden,hidden,batch_first=True); self.head=nn.Linear(hidden,horizon)
    def forward(self,x):
        node_features=x.permute(0,1,3,2); hidden=torch.relu(self.projection(node_features)); adjacency=torch.softmax(self.adaptive_logits,dim=-1); hidden=torch.einsum("ij,btjf->btif",adjacency,hidden); batch,timesteps,nodes,width=hidden.shape; sequence=hidden.permute(0,2,1,3).reshape(batch*nodes,timesteps,width); encoded,_=self.gru(sequence); prediction=self.head(encoded[:,-1]).reshape(batch,nodes,self.horizon); return prediction.permute(0,2,1)

class STGCN(nn.Module):
    """Compact spatio-temporal graph convolution baseline."""
    def __init__(self,features:int,nodes:int,horizon:int,hidden:int=64):
        super().__init__(); self.nodes=nodes; self.horizon=horizon
        self.temporal=nn.Sequential(nn.Conv1d(features,hidden,3,padding=1),nn.ReLU(),nn.Conv1d(hidden,hidden,3,padding=1),nn.ReLU())
        self.graph=nn.Linear(hidden,hidden); self.head=nn.Linear(hidden,horizon)
        self.adaptive_logits=nn.Parameter(torch.zeros(nodes,nodes))
    def forward(self,x):
        batch,timesteps,features,nodes=x.shape
        sequence=x.permute(0,3,2,1).reshape(batch*nodes,features,timesteps)
        encoded=self.temporal(sequence).transpose(1,2)
        encoded=self.graph(encoded)
        encoded=encoded[:,-1].reshape(batch,nodes,-1)
        adjacency=torch.softmax(self.adaptive_logits,dim=-1)
        encoded=torch.einsum("ij,bjf->bif",adjacency,encoded)
        return self.head(encoded).permute(0,2,1)

class DCRNN(nn.Module):
    """Diffusion-style recurrent baseline with a learned directed graph."""
    def __init__(self,features:int,nodes:int,horizon:int,hidden:int=64):
        super().__init__(); self.nodes=nodes; self.horizon=horizon
        self.input=nn.Linear(features,hidden); self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.head=nn.Linear(hidden,horizon); self.diffusion_logits=nn.Parameter(torch.zeros(nodes,nodes))
    def forward(self,x):
        batch,timesteps,features,nodes=x.shape
        hidden=torch.relu(self.input(x.permute(0,3,1,2)))
        encoded,_=self.gru(hidden.reshape(batch*nodes,timesteps,-1))
        encoded=encoded[:,-1].reshape(batch,nodes,-1)
        adjacency=torch.softmax(self.diffusion_logits,dim=-1)
        encoded=torch.einsum("ij,bjf->bif",adjacency,encoded)
        return self.head(encoded).permute(0,2,1)

class GraphWaveNet(nn.Module):
    """Graph WaveNet-style temporal convolution with adaptive adjacency."""
    def __init__(self,features:int,nodes:int,horizon:int,hidden:int=64):
        super().__init__(); self.nodes=nodes; self.horizon=horizon
        self.temporal=nn.Sequential(nn.Conv1d(features,hidden,2),nn.GELU(),nn.Conv1d(hidden,hidden,2),nn.GELU())
        self.head=nn.Linear(hidden,horizon); self.node_embeddings=nn.Parameter(torch.randn(nodes,hidden)*0.05)
    def forward(self,x):
        batch,timesteps,features,nodes=x.shape
        sequence=x.permute(0,3,2,1).reshape(batch*nodes,features,timesteps)
        encoded=self.temporal(sequence)[:,:,-1].reshape(batch,nodes,-1)
        adjacency=torch.softmax(torch.relu(self.node_embeddings@self.node_embeddings.T),dim=-1)
        encoded=encoded+torch.einsum("ij,bjf->bif",adjacency,encoded)
        return self.head(encoded).permute(0,2,1)

class AGCRN(nn.Module):
    """Adaptive graph convolution recurrent network baseline."""
    def __init__(self,features:int,nodes:int,horizon:int,hidden:int=64):
        super().__init__(); self.nodes=nodes; self.horizon=horizon
        self.node_embeddings=nn.Parameter(torch.randn(nodes,hidden)*0.05)
        self.input=nn.Linear(features,hidden); self.gru=nn.GRU(hidden,hidden,batch_first=True); self.head=nn.Linear(hidden,horizon)
    def forward(self,x):
        batch,timesteps,features,nodes=x.shape
        hidden=torch.relu(self.input(x.permute(0,3,1,2)))
        hidden=hidden+ self.node_embeddings[None,:,None,:]
        encoded,_=self.gru(hidden.reshape(batch*nodes,timesteps,-1))
        encoded=encoded[:,-1].reshape(batch,nodes,-1)
        adjacency=torch.softmax(torch.relu(self.node_embeddings@self.node_embeddings.T),dim=-1)
        encoded=torch.einsum("ij,bjf->bif",adjacency,encoded)
        return self.head(encoded).permute(0,2,1)

class ARIMABaseline:
    """A dependency-free per-node differenced AR(1) statistical baseline."""
    def __init__(self,horizon:int): self.horizon=horizon; self.intercepts=None; self.coefficients=None
    def fit(self,x):
        speed=x[:,:,0,:].detach().cpu().numpy()
        difference=np.diff(speed,axis=1)
        self.intercepts=difference.mean(axis=(0,1))
        centered=difference-self.intercepts
        denominator=(centered[:,:-1]**2).sum(axis=(0,1))+1e-8
        self.coefficients=(centered[:,1:]*centered[:,:-1]).sum(axis=(0,1))/denominator
        return self
    def predict(self,x):
        if self.intercepts is None: self.fit(x)
        previous=x[:,-1,0,:].detach().cpu().numpy()
        last_difference=np.zeros_like(previous)
        outputs=[]
        for _ in range(self.horizon):
            last_difference=self.intercepts+self.coefficients*last_difference
            previous=previous+last_difference
            outputs.append(previous.copy())
        return torch.from_numpy(np.stack(outputs, axis=1)).to(dtype=x.dtype, device=x.device)

class Persistence:
    def __init__(self,horizon:int): self.horizon=horizon
    def predict(self,x): return x[:,-1,0,:].unsqueeze(1).repeat(1,self.horizon,1)
