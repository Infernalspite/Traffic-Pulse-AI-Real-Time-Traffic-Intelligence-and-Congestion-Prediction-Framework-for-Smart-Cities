"""PyTorch traffic forecasting competitors sharing one tensor contract."""
from __future__ import annotations
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

class Persistence:
    def __init__(self,horizon:int): self.horizon=horizon
    def predict(self,x): return x[:,-1,0,:].unsqueeze(1).repeat(1,self.horizon,1)
