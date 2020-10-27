"""
    Model definitions.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

# GNN AE using EdgeConv (mean aggregation graph operation). Basic GAE model.
class EdgeNet(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNet, self).__init__()
        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, hidden_dim),
                               nn.ReLU(),
        )
        
        decoder_nn = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, input_dim)
        )
        
        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder = EdgeConv(nn=encoder_nn,aggr=aggr)
        self.decoder = EdgeConv(nn=decoder_nn,aggr=aggr)

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.encoder(data.x,data.edge_index)
        data.x = self.decoder(data.x,data.edge_index)
        return data.x
    
# GVAE based on EdgeNet model above.
class EdgeNetVAE(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetVAE, self).__init__()
        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(big_dim, hidden_dim)
        self.var_layer = nn.Linear(big_dim, hidden_dim)
        
        decoder_nn = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, input_dim)
        )
        
        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder = EdgeConv(nn=encoder_nn,aggr=aggr)
        self.decoder = EdgeConv(nn=decoder_nn,aggr=aggr)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.encoder(data.x,data.edge_index)
        mu = self.mu_layer(data.x)
        log_var = self.var_layer(data.x)
        z = self.reparameterize(mu, log_var)
        data.x = self.decoder(z,data.edge_index)
        return data.x, mu, log_var

# All GNN MetaLayer components (GAE with global and edge features)
class EdgeEncoder(torch.nn.Module):
    def __init__(self):
        super(EdgeEncoder, self).__init__()
        self.edge_mlp = Seq(Lin(4+4, 32), 
                            ReLU(),
                            Lin(32, 32))

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest], 1)
        return self.edge_mlp(out)

class NodeEncoder(torch.nn.Module):
    def __init__(self):
        super(NodeEncoder, self).__init__()
        self.node_mlp_1 = Seq(Lin(4+32, 32), 
                              ReLU(), 
                              Lin(32, 32))
        self.node_mlp_2 = Seq(Lin(4+32, 32), 
                              ReLU(), 
                              Lin(32, 2))

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

class GlobalEncoder(torch.nn.Module):
    def __init__(self):
        super(GlobalEncoder, self).__init__()
        self.global_mlp = Seq(Lin(2, 32), 
                              ReLU(), 
                              Lin(32, 32))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = scatter_mean(x, batch, dim=0)
        return self.global_mlp(out)


class EdgeDecoder(torch.nn.Module):
    def __init__(self):
        super(EdgeDecoder, self).__init__()
        self.edge_mlp = Seq(Lin(2+2+32, 32), 
                            ReLU(), 
                            Lin(32, 32))

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, u[batch]], 1)
        return self.edge_mlp(out)

class NodeDecoder(torch.nn.Module):
    def __init__(self):
        super(NodeDecoder, self).__init__()
        self.node_mlp_1 = Seq(Lin(2+32, 32), 
                              ReLU(), 
                              Lin(32, 32))
        self.node_mlp_2 = Seq(Lin(2+32+32, 32), 
                              ReLU(), 
                              Lin(32, 4))

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

class GNNAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(GNNAutoEncoder, self).__init__()
        self.encoder = MetaLayer(EdgeEncoder(), NodeEncoder(), GlobalEncoder())
        self.decoder = MetaLayer(EdgeDecoder(), NodeDecoder(), None)
    
    def forward(self, data):
        x, edge_attr, u = self.encoder(data.x, data.edge_index, None, None, data.batch)
        x, edge_attr, u = self.decoder(x, data.edge_index, None, u, data.batch)
        return x
