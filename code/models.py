"""
    Model definitions.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool, DynamicEdgeConv
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

# no EdgeConvs
class AE(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear((input_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, hidden_dim),
                               nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(nn.Linear((hidden_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, input_dim)
        )
        
        self.batchnorm = nn.BatchNorm1d(input_dim)

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.encoder(data.x)
        data.x = self.decoder(data.x)
        return data.x

# Double EdgeConv for encoder + decoder
class EdgeNetDeeper(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetDeeper, self).__init__()

        encoder_nn_1 = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
        )
        encoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, hidden_dim),
                                   nn.ReLU(),
        )
        decoder_nn_1 = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU()
        )
        decoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, input_dim)
        )

        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)
        self.decoder_2 = EdgeConv(nn=decoder_nn_2,aggr=aggr)

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.encoder_1(data.x,data.edge_index)
        data.x = self.encoder_2(data.x,data.edge_index)
        data.x = self.decoder_1(data.x,data.edge_index)
        data.x = self.decoder_2(data.x,data.edge_index)
        return data.x

# 2 EdgeConv Wider
class EdgeNetDeeper2(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetDeeper2, self).__init__()

        encoder_nn_1 = nn.Sequential(nn.Linear(2*(input_dim), big_dim*2),
                                   nn.ReLU(),
                                   nn.Linear(big_dim*2, big_dim*2),
                                   nn.ReLU(),
                                   nn.Linear(big_dim*2, big_dim),
                                   nn.ReLU(),
        )
        encoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, hidden_dim),
                                   nn.ReLU(),
        )
        decoder_nn_1 = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim*2),
                                   nn.ReLU()
        )
        decoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim*2), big_dim*2),
                                   nn.ReLU(),
                                   nn.Linear(big_dim*2, big_dim*2),
                                   nn.ReLU(),
                                   nn.Linear(big_dim*2, input_dim)
        )

        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)
        self.decoder_2 = EdgeConv(nn=decoder_nn_2,aggr=aggr)

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.encoder_1(data.x,data.edge_index)
        data.x = self.encoder_2(data.x,data.edge_index)
        data.x = self.decoder_1(data.x,data.edge_index)
        data.x = self.decoder_2(data.x,data.edge_index)
        return data.x

# 3 EdgeConv Wider symmetrical encoder/decoder
class EdgeNetDeeper3(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetDeeper3, self).__init__()

        encoder_nn_1 = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
        )
        encoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
        )
        encoder_nn_3 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, hidden_dim),
                                   nn.ReLU(),
        )
        decoder_nn_1 = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU()
        )
        decoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU()
        )
        decoder_nn_3 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, input_dim)
        )

        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.encoder_3 = EdgeConv(nn=encoder_nn_3,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)
        self.decoder_2 = EdgeConv(nn=decoder_nn_2,aggr=aggr)
        self.decoder_3 = EdgeConv(nn=decoder_nn_3,aggr=aggr)

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.encoder_1(data.x,data.edge_index)
        data.x = self.encoder_2(data.x,data.edge_index)
        data.x = self.encoder_3(data.x,data.edge_index)
        data.x = self.decoder_1(data.x,data.edge_index)
        data.x = self.decoder_2(data.x,data.edge_index)
        data.x = self.decoder_3(data.x,data.edge_index)
        return data.x
    
# 2 EdgeConv Encoder, 1 EdgeConv decoder and thinner
class EdgeNetDeeper4(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetDeeper4, self).__init__()

        encoder_nn_1 = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
        )
        encoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim // 2),
                                   nn.ReLU(),
                                   nn.Linear(big_dim // 2, big_dim // 2),
                                   nn.ReLU(),
                                   nn.Linear(big_dim // 2, hidden_dim),
                                   nn.ReLU(),
        )
        decoder_nn_1 = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, input_dim)
        )

        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.encoder_1(data.x,data.edge_index)
        data.x = self.encoder_2(data.x,data.edge_index)
        data.x = self.decoder_1(data.x,data.edge_index)
        return data.x

# Baseline Edgenet but deeper encoder/decoder
class EdgeNetDeeper5(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetDeeper5, self).__init__()
        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
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

# Deeper vers + more Batchnorm
class EdgeNetDeeperBN(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetDeeperBN, self).__init__()

        encoder_nn_1 = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim)
        )
        encoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim)
        )
        decoder_nn_1 = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim)
        )
        decoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, input_dim)
        )

        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)
        self.decoder_2 = EdgeConv(nn=decoder_nn_2,aggr=aggr)

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.encoder_1(data.x,data.edge_index)
        data.x = self.encoder_2(data.x,data.edge_index)
        data.x = self.decoder_1(data.x,data.edge_index)
        data.x = self.decoder_2(data.x,data.edge_index)
        return data.x

# 2 EdgeConv Wider
class EdgeNetDeeper2BN(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetDeeper2BN, self).__init__()

        encoder_nn_1 = nn.Sequential(nn.Linear(2*(input_dim), big_dim*2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, big_dim*2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim)
        )
        encoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim)
        )
        decoder_nn_1 = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.BatchNorm1d(big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim*2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim*2)
        )
        decoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim*2), big_dim*2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, big_dim*2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, input_dim)
        )

        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)
        self.decoder_2 = EdgeConv(nn=decoder_nn_2,aggr=aggr)

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.encoder_1(data.x,data.edge_index)
        data.x = self.encoder_2(data.x,data.edge_index)
        data.x = self.decoder_1(data.x,data.edge_index)
        data.x = self.decoder_2(data.x,data.edge_index)
        return data.x

# 3 EdgeConv Wider symmetrical encoder/decoder
class EdgeNetDeeper3BN(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetDeeper3BN, self).__init__()

        encoder_nn_1 = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim)
        )
        encoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim)
        )
        encoder_nn_3 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim)
        )
        decoder_nn_1 = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim)
        )
        decoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim)
        )
        decoder_nn_3 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, input_dim)
        )

        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.encoder_3 = EdgeConv(nn=encoder_nn_3,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)
        self.decoder_2 = EdgeConv(nn=decoder_nn_2,aggr=aggr)
        self.decoder_3 = EdgeConv(nn=decoder_nn_3,aggr=aggr)

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.encoder_1(data.x,data.edge_index)
        data.x = self.encoder_2(data.x,data.edge_index)
        data.x = self.encoder_3(data.x,data.edge_index)
        data.x = self.decoder_1(data.x,data.edge_index)
        data.x = self.decoder_2(data.x,data.edge_index)
        data.x = self.decoder_3(data.x,data.edge_index)
        return data.x
    
# 2 EdgeConv Encoder, 1 EdgeConv decoder and thinner
class EdgeNetDeeper4BN(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetDeeper4BN, self).__init__()

        encoder_nn_1 = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim)
        )
        encoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim // 2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim // 2),
                                   nn.Linear(big_dim // 2, big_dim // 2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim // 2),
                                   nn.Linear(big_dim // 2, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim)
        )
        decoder_nn_1 = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, input_dim)
        )

        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.encoder_1(data.x,data.edge_index)
        data.x = self.encoder_2(data.x,data.edge_index)
        data.x = self.decoder_1(data.x,data.edge_index)
        return data.x

# Baseline Edgenet but deeper encoder/decoder
class EdgeNetDeeper5BN(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetDeeper5BN, self).__init__()
        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.BatchNorm1d(big_dim),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.BatchNorm1d(big_dim),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.BatchNorm1d(big_dim),
                               nn.Linear(big_dim, hidden_dim),
                               nn.ReLU(),
                               nn.BatchNorm1d(hidden_dim)
        )
        
        decoder_nn = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                               nn.ReLU(),
                               nn.BatchNorm1d(big_dim),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.BatchNorm1d(big_dim),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.BatchNorm1d(big_dim),
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

# GNN AE using EdgeConv (mean aggregation graph operation) and node embedding.
class EdgeNetEmbed(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetEmbed, self).__init__()
        self.embed_nn = nn.Sequential(nn.Linear(input_dim, big_dim),
                                      nn.ReLU(),
                                      nn.Linear(big_dim, big_dim),
                                      nn.ReLU(),
                                      nn.Linear(big_dim, big_dim),
                                      nn.ReLU()
        )                                
        encoder_nn = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
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
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU()
                                   
        )
        self.deembed_nn = nn.Sequential(nn.Linear(big_dim, big_dim),
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
        data.x = self.embed_nn(data.x)
        data.x = self.encoder(data.x,data.edge_index)
        data.x = self.decoder(data.x,data.edge_index)
        data.x = self.deembed_nn(data.x)
        return data.x

# Using Dynamic Edge Convolution
class EdgeNetDynamic(torch.nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetDynamic, self).__init__()
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

        self.encoder = DynamicEdgeConv(nn=encoder_nn,aggr=aggr,k=3)
        self.decoder = DynamicEdgeConv(nn=decoder_nn,aggr=aggr,k=3)
    
    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.encoder(data.x,data.batch)
        data.x = self.decoder(data.x,data.batch)
        return data.x

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

# models
model_list = {
    0: EdgeNet, 
    1: EdgeNetDeeper,
    2: EdgeNetDeeper2,
    3: EdgeNetDeeper3,
    4: EdgeNetDeeper4,
    5: EdgeNetDeeper5,
    6: AE,
    7: EdgeNetVAE,
    8: EdgeNetEmbed,
    9: GNNAutoEncoder,
    10: EdgeNetDeeperBN,
    11: EdgeNetDeeper2BN,
    12: EdgeNetDeeper3BN,
    13: EdgeNetDeeper4BN,
    14: EdgeNetDeeper5BN
}