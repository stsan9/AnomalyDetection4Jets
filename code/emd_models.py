import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import DynamicEdgeConv, EdgeConv, global_mean_pool
from torch_scatter import scatter_mean

class DeeperDynamicEdgeNet(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, bigger_dim=256, global_dim=2, output_dim=1, k=16, aggr='mean'):
        super(DeeperDynamicEdgeNet, self).__init__()
        convnn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.BatchNorm1d(big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.BatchNorm1d(big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
        )
        convnn2 = nn.Sequential(nn.Linear(2*(big_dim+input_dim), big_dim*2),
                                nn.BatchNorm1d(big_dim*2),
                                nn.ReLU(),
                                nn.Linear(big_dim*2, big_dim*2),
                                nn.BatchNorm1d(big_dim*2),
                                nn.ReLU(),
                                nn.Linear(big_dim*2, big_dim*2),
        )
        convnn3 = nn.Sequential(nn.Linear(2*(big_dim*2+input_dim), big_dim*4),
                                nn.BatchNorm1d(big_dim*4),
                                nn.ReLU(),
                                nn.Linear(big_dim*4, big_dim*4),
                                nn.BatchNorm1d(big_dim*4),
                                nn.ReLU(),
                                nn.Linear(big_dim*4, big_dim*4),
        )
                
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.batchnormglobal = nn.BatchNorm1d(global_dim)
        self.outnn = nn.Sequential(nn.Linear(big_dim*4+input_dim+global_dim, bigger_dim),
                                   nn.BatchNorm1d(bigger_dim),
                                   nn.ReLU(),
                                   nn.Linear(bigger_dim, bigger_dim),
                                   nn.BatchNorm1d(bigger_dim),
                                   nn.ReLU(),
                                   nn.Linear(bigger_dim, output_dim)
        )
        
        self.conv = DynamicEdgeConv(nn=convnn, aggr=aggr, k=k)
        self.conv2 = DynamicEdgeConv(nn=convnn2, aggr=aggr, k=k)
        self.conv3 = DynamicEdgeConv(nn=convnn3, aggr=aggr, k=k)

    def forward(self, data):
        x1 = self.batchnorm(data.x)        
        x2 = self.conv(data.x, data.batch)
        x = torch.cat([x1, x2],dim=-1)
        x2 = self.conv2(x, data.batch)
        x = torch.cat([x1, x2],dim=-1)
        x2 = self.conv3(x, data.batch)
        x = torch.cat([x1, x2],dim=-1)
        u1 = self.batchnormglobal(data.u)
        u2 = scatter_mean(x, data.batch, dim=0)
        data.u = torch.cat([u1, u2],dim=-1)       
        return self.outnn(data.u)

class EmdNN(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, bigger_dim=128, global_dim=2, output_dim=1, k=16, aggr='mean', device='cuda:0'):
        super(EmdNN, self).__init__()
        self.EdgeNet = DeeperDynamicEdgeNet(input_dim, big_dim, bigger_dim, global_dim, output_dim, k, aggr).to(device)

    def forward(self, data):
        # dual copies with different orderings
        data_1 = data
        data_2 = data.clone()
        data_2.x[:,-1] *= -1

        emd_1 = self.EdgeNet(data_1)
        emd_2 = self.EdgeNet(data_2)
        loss = (emd_1 + emd_2) / 2
        return loss, emd_1, emd_2

class EmdNNSpl(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, bigger_dim=128, global_dim=2, output_dim=1, k=16, aggr='mean', device='cuda:0'):
        super(EmdNNSpl, self).__init__()
        self.EdgeNet = DeeperDynamicEdgeNet(input_dim, big_dim, bigger_dim, global_dim, output_dim, k, aggr).to(device)

    def forward(self, data):
        # dual copies with different orderings
        data_1 = data
        data_2 = data.clone()
        data_2.x[:,-1] *= -1

        spl = nn.Softplus()
        emd_1 = spl(self.EdgeNet(data_1))
        emd_2 = spl(self.EdgeNet(data_2))
        loss = (emd_1 + emd_2) / 2
        return loss, emd_1, emd_2

class EmdNNRel(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, bigger_dim=128, global_dim=2, output_dim=1, k=16, aggr='mean', device='cuda:0'):
        super(EmdNNRel, self).__init__()
        self.EdgeNet = DeeperDynamicEdgeNet(input_dim, big_dim, bigger_dim, global_dim, output_dim, k, aggr).to(device)

    def forward(self, data):
        # dual copies with different orderings
        data_1 = data
        data_2 = data.clone()
        data_2.x[:,-1] *= -1

        rel = nn.ReLU()
        emd_1 = rel(self.EdgeNet(data_1))
        emd_2 = rel(self.EdgeNet(data_2))
        loss = (emd_1 + emd_2) / 2
        return loss, emd_1, emd_2
