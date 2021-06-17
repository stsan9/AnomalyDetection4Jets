"""
    Model definitions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from loss_util import get_ptetaphi, load_emd_model
from torch_scatter import scatter_mean, scatter
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import EdgeConv, global_mean_pool, DynamicEdgeConv
from torch_geometric.nn import EdgeConv, global_mean_pool

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
        x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        x = self.decoder(x,data.edge_index)
        return x

class EdgeNetEMD(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean', emd_modname='EmdNNRel.best.pth'):
        super(EdgeNetEMD, self).__init__()
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

        self.emd_model = load_emd_model(emd_modname, device='cuda' if torch.cuda.is_available() else 'cpu')

    def emd_loss(self, x, y, batch):
        self.emd_model.eval()
        device = x.device.type
        try:
            x = get_ptetaphi(x, batch)
            y = get_ptetaphi(y, batch)
        except ValueError as e:
            print('Error:', e)
            raise RuntimeError('emd_loss had error') from e
        # concatenate column of 1s to one jet and -1 to other jet
        x = torch.cat((x,torch.ones(len(x),1).to(device)), 1)
        y = torch.cat((y,torch.ones(len(y),1).to(device)*-1), 1)
        # normalize pt
        Ex = scatter(src=x[:,0],index=batch)
        Ey = scatter(src=y[:,0],index=batch)
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        Ex_repeat = torch.repeat_interleave(Ex, counts, dim=0)
        Ey_repeat = torch.repeat_interleave(Ey, counts, dim=0)
        x[:,0] = x[:,0].clone() / Ex_repeat
        y[:,0] = y[:,0].clone() / Ey_repeat
        # create data object for emd model
        jet_pair = torch.cat((x,y),0)
        u = torch.cat((Ex.view(-1,1),Ey.view(-1,1)),dim=1) / 100.0
        data = Data(x=jet_pair, batch=torch.cat((batch,batch)), u=u).to(device)
        # get emd between x and y
        out = self.emd_model(data)
        emd = out[0]    # ignore other model outputs
        return emd

    def forward(self, data):
        x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        x = self.decoder(x,data.edge_index)
        print(x == data.x)
        loss = self.emd_loss(x, data.x, data.batch)
        return x, loss
    
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
        x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        mu = self.mu_layer(x)
        log_var = self.var_layer(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z,data.edge_index)
        return x, mu, log_var

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
        x = self.batchnorm(data.x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
        x = self.batchnorm(data.x)
        x = self.encoder_1(x,data.edge_index)
        x = self.encoder_2(x,data.edge_index)
        x = self.decoder_1(x,data.edge_index)
        x = self.decoder_2(x,data.edge_index)
        return x

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
        x = self.batchnorm(data.x)
        x = self.encoder_1(x,data.edge_index)
        x = self.encoder_2(x,data.edge_index)
        x = self.decoder_1(x,data.edge_index)
        x = self.decoder_2(x,data.edge_index)
        return x

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
        x = self.batchnorm(data.x)
        x = self.encoder_1(x,data.edge_index)
        x = self.encoder_2(x,data.edge_index)
        x = self.encoder_3(x,data.edge_index)
        x = self.decoder_1(x,data.edge_index)
        x = self.decoder_2(x,data.edge_index)
        x = self.decoder_3(x,data.edge_index)
        return x
    
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
        x = self.batchnorm(data.x)
        x = self.encoder_1(x,data.edge_index)
        x = self.encoder_2(x,data.edge_index)
        x = self.decoder_1(x,data.edge_index)
        return x

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
        x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        x = self.decoder(x,data.edge_index)
        return x

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
        x = self.batchnorm(data.x)
        x = self.encoder_1(x,data.edge_index)
        x = self.encoder_2(x,data.edge_index)
        x = self.decoder_1(x,data.edge_index)
        x = self.decoder_2(x,data.edge_index)
        return x

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
        x = self.batchnorm(data.x)
        x = self.encoder_1(x,data.edge_index)
        x = self.encoder_2(x,data.edge_index)
        x = self.decoder_1(x,data.edge_index)
        x = self.decoder_2(x,data.edge_index)
        return x

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
        x = self.batchnorm(data.x)
        x = self.encoder_1(x,data.edge_index)
        x = self.encoder_2(x,data.edge_index)
        x = self.encoder_3(x,data.edge_index)
        x = self.decoder_1(x,data.edge_index)
        x = self.decoder_2(x,data.edge_index)
        x = self.decoder_3(x,data.edge_index)
        return x
    
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
        x = self.batchnorm(data.x)
        x = self.encoder_1(x,data.edge_index)
        x = self.encoder_2(x,data.edge_index)
        x = self.decoder_1(x,data.edge_index)
        return x

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
        x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        x = self.decoder(x,data.edge_index)
        return x

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
        x = self.batchnorm(data.x)
        x = self.embed_nn(x)
        x = self.encoder(x,data.edge_index)
        x = self.decoder(x,data.edge_index)
        x = self.deembed_nn(x)
        return x

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
        x = self.batchnorm(data.x)
        x = self.encoder(x,data.batch)
        x = self.decoder(x,data.batch)
        return x

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

class MetaLayerGAE(torch.nn.Module):
    def __init__(self):
        super(MetaLayerGAE, self).__init__()
        self.encoder = MetaLayer(EdgeEncoder(), NodeEncoder(), GlobalEncoder())
        self.decoder = MetaLayer(EdgeDecoder(), NodeDecoder(), None)
    
    def forward(self, data):
        x, edge_attr, u = self.encoder(data.x, data.edge_index, None, None, data.batch)
        x, edge_attr, u = self.decoder(x, data.edge_index, None, u, data.batch)
        return x
