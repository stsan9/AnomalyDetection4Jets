import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data, DataListLoader, Batch
import os.path as osp
import numpy as np
import math
import matplotlib.pyplot as plt
from graph_data import GraphDataset
from torch.utils.data import random_split

# define model
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
    
    def forward(self, x, edge_index, batch):
        x, edge_attr, u = self.encoder(x, edge_index, None, None, batch)
        x, edge_attr, u = self.decoder(x, edge_index, None, u, batch)
        return x

# load in data and define dimensions
gdata = GraphDataset(root='/anomalyvol/data/gnn_node_global_merge', bb=0)

input_dim = 4
big_dim = 32
hidden_dim = 2
fulllen = len(gdata)
tv_frac = 0.10
tv_num = math.ceil(fulllen*tv_frac)
splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
batch_size = 4
n_epochs = 800
lr = 0.001
patience = 10
device = 'cuda:0'
model_fname = 'GNN_node_global'

model = GNNAutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

def collate(items):
    l = sum(items, [])
    return Batch.from_data_list(l)

# establish train, valid, and test datasets/loaders
torch.manual_seed(0)
train_dataset, valid_dataset, test_dataset = random_split(gdata, [fulllen-2*tv_num,tv_num,tv_num])
train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
train_loader.collate_fn = collate
valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
valid_loader.collate_fn = collate
test_loader = DataListLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
test_loader.collate_fn = collate
train_samples = len(train_dataset)
valid_samples = len(valid_dataset)
test_samples = len(test_dataset)

@torch.no_grad()
def test(model,loader,total,batch_size):
    model.eval()
    
    mse = nn.MSELoss(reduction='mean')

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        y = data.x # the model will overwrite data.x
        batch_output = model(data.x, data.edge_index, data.batch)
        batch_loss_item = mse(batch_output, y).item()
        sum_loss += batch_loss_item
        t.set_description("loss = %.5f" % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def train(model, optimizer, loader, total, batch_size):
    model.train()
    
    mse = nn.MSELoss(reduction='mean')

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        y = data.x # the model will overwrite data.x
        optimizer.zero_grad()
        batch_output = model(data.x, data.edge_index, data.batch)
        batch_loss = mse(batch_output, y)
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description("loss = %.5f" % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()
    
    return sum_loss/(i+1)

modpath = osp.join('/anomalyvol/models/gnn/',model_fname+'.best.pth')
try:
    model.load_state_dict(torch.load(modpath))
except:
    pass

stale_epochs = 0
best_valid_loss = 99999
for epoch in range(0, n_epochs):
    loss = train(model, optimizer, train_loader, train_samples, batch_size)
    valid_loss = test(model, valid_loader, valid_samples, batch_size)
    print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
    print('               Validation Loss: {:.4f}'.format(valid_loss))

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        modpath = osp.join('/anomalyvol/models/gnn_node_global/',model_fname+'.best.pth')
        print('New best model saved to:',modpath)
        torch.save(model.state_dict(),modpath)
        stale_epochs = 0
    else:
        print('Stale epoch')
        stale_epochs += 1
    if stale_epochs >= patience:
        print('Early stopping after %i stale epochs'%patience)
        break
        
print("Completed")