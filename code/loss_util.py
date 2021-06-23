import sys
import torch
import emd_models
import numpy as np
import torch_scatter
import os.path as osp
from torch_geometric.data import Data

multi_gpu = torch.cuda.device_count()>1
eps = 1e-12

def load_emd_model(modname, device):
    emd_model = getattr(emd_models, modname[:-9])(device=device)
    modpath = osp.join('/anomalyvol/emd_models/', modname)
    emd_model.load_state_dict(torch.load(modpath, map_location=torch.device(device)))
    return emd_model

def arctanh(x):
    return torch.log1p(2*x/(1-x + eps)) / 2

def get_ptetaphi(x,batch):
    px = x[:,0]
    py = x[:,1]
    pz = x[:,2]
    p = torch.sqrt(torch.square(px) + torch.square(py) + torch.square(pz) + eps)
    pt = torch.sqrt(torch.square(px) + torch.square(py) + eps)
    eta = arctanh(pz / (p + eps))
    phi = torch.atan(py / (px + eps))
    ts = [px,py,pz,p,pt,eta,phi]
    for e in ts:
        if True in torch.isnan(e):
            raise ValueError('nan in get_ptetaphi')
    mat = torch.stack((pt,eta,phi),dim=1)
    return mat

def preprocess_emdnn_input(x, y, batch):
    # px py pz -> pt eta phi
    x = get_ptetaphi(x, batch)
    y = get_ptetaphi((y+eps), batch)

    # center by pt centroid while accounting for torch geo batching
    _, counts = torch.unique_consecutive(batch, return_counts=True)
    n = torch_scatter.scatter(x[:,1:3].clone() * x[:,0,None].clone(), batch, dim=0, reduce='sum')
    d = torch_scatter.scatter(y[:,0], batch, dim=0, reduce='sum')
    yphi_avg = (n.T / (d + eps)).T  # returns yphi_avg for each batch
    yphi_avg = torch.repeat_interleave(yphi_avg, counts, dim=0)
    x[:,1:3] -= yphi_avg

    n = torch_scatter.scatter(y[:,1:3].clone() * y[:,0,None].clone(), batch, dim=0, reduce='sum')
    d = torch_scatter.scatter(y[:,0], batch, dim=0, reduce='sum')
    yphi_avg = (n.T / (d + eps)).T  # returns yphi_avg for each batch
    yphi_avg = torch.repeat_interleave(yphi_avg, counts, dim=0)
    y[:,1:3] -= yphi_avg
    y = y + eps
 
    # normalize pt
    Ex = torch_scatter.scatter(src=x[:,0],index=batch, reduce='sum')
    Ey = torch_scatter.scatter(src=y[:,0],index=batch, reduce='sum')
    Ex_repeat = torch.repeat_interleave(Ex, counts, dim=0)
    Ey_repeat = torch.repeat_interleave(Ey, counts, dim=0)
    x[:,0] = x[:,0].clone() / (Ex_repeat + eps)
    y[:,0] = y[:,0].clone() / (Ey_repeat + eps)

    device = x.device.type
    x = torch.cat((x,torch.ones(len(x),1).to(device)), 1)
    y = torch.cat((y,torch.ones(len(y),1).to(device)*-1), 1)
    jet_pair = torch.cat((x,y),0)
    u = torch.cat((Ex.view(-1,1),Ey.view(-1,1)),dim=1) / 100.0
    data = Data(x=jet_pair, batch=torch.cat((batch,batch)), u=u).to(self.device)
    return data

def pairwise_distance(x, y, device=None):
    if (x.shape[0] != y.shape[0]):
        raise ValueError(f"The batch size of x and y are not equal! x.shape[0] is {x.shape[0]}, whereas y.shape[0] is {y.shape[0]}!")
    if (x.shape[-1] != y.shape[-1]):
        raise ValueError(f"Feature dimension of x and y are not equal! x.shape[-1] is {x.shape[-1]}, whereas y.shape[-1] is {y.shape[-1]}!")

    if device is None:
        device = x.device

    batch_size = x.shape[0]
    num_row = x.shape[1]
    num_col = y.shape[1]
    vec_dim = x.shape[-1]

    x1 = x.repeat(1, 1, num_col).view(batch_size, -1, num_col, vec_dim).to(device)
    y1 = y.repeat(1, num_row, 1).view(batch_size, num_row, -1, vec_dim).to(device)

    dist = torch.norm(x1 - y1 + eps, dim=-1)

    return dist

class LossFunction:
    def __init__(self, lossname, emd_modname='EmdNNRel.best.pth', device='cuda:0'):
        if lossname == 'mse':
            loss = torch.nn.MSELoss(reduction='mean')
        else:
            loss = getattr(self, lossname)
            if lossname == 'emd_loss' and not multi_gpu:
                # keep emd model in memory
                # if using DataParallel it's merged into the network's forward pass to distribute gpu memory
                emd_model = load_emd_model(emd_modname,device)
                self.emd_model = emd_model.requires_grad_(False)
        self.name = lossname
        self.loss_ftn = loss
        self.device = device

    def chamfer_loss(self, x, y, batch):
        x = get_ptetaphi(x, batch)
        y = get_ptetaphi(y, batch) 

        # https://github.com/zichunhao/mnist_graph_autoencoder/blob/master/utils/loss.py
        dist = pairwise_distance(x, y, self.device)

        min_dist_xy = torch.min(dist, dim=-1)
        min_dist_yx = torch.min(dist, dim=-2)  # Equivalent to permute the last two axis

        loss = torch.sum(min_dist_xy.values + min_dist_yx.values)

        return loss

    # Reconstruction + KL divergence losses
    def vae_loss(self, x, y, mu, logvar):
        BCE = chamfer_loss(x,y)
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def emd_loss(self, x, y, batch):
        self.emd_model.eval()
        try:
            data = preprocess_emdnn_input(x, y, batch)
        except ValueError as e:
            print('Error:', e)
            raise RuntimeError('emd_loss had error') from e
        out = self.emd_model(data)
        emd = out[0]
        return emd
