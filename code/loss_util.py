import sys
import torch
import emd_models
import numpy as np
import torch_scatter
import os.path as osp
from torch_geometric.data import Data
from emd_loss import emd_loss as deepemd
from torch_geometric.utils import to_dense_batch

multi_gpu = torch.cuda.device_count()>1
eps = 1e-12
torch.autograd.set_detect_anomaly(True)

def load_emd_model(modname, device):
    emd_model = getattr(emd_models, modname[:-9])(device=device)
    modpath = osp.join('/anomalyvol/emd_models/', modname)
    emd_model.load_state_dict(torch.load(modpath, map_location=device))
    return emd_model

def preprocess_emdnn_input(x, y, batch):
    # center by pt centroid while accounting for torch geo batching
    _, unique_batches = torch.unique_consecutive(batch, return_counts=True)
 
    device = x.device.type
    x = torch.cat((x, torch.ones((len(x),1), device=device)), 1)
    y = torch.cat((y, torch.ones((len(y),1), device=device) * -1), 1)
    jet_pair = torch.cat((x,y), 0)
    data = Data(x=jet_pair, batch=torch.cat((batch,batch))).to(device)
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
    def __init__(self, lossname, emd_modname='EmdNNRel.best.pth', device=torch.device('cuda:0')):
        if lossname == 'mse':
            loss = torch.nn.MSELoss(reduction='mean')
        else:
            loss = getattr(self, lossname)
            if lossname == 'emd_loss':
                # if using DataParallel it's merged into the network's forward pass to distribute gpu memory
                self.emd_model = load_emd_model(emd_modname, device)
                # self.emd_model = emd_model.requires_grad_(False)
        self.name = lossname
        self.loss_ftn = loss
        self.device = device

    def chamfer_loss(self, x, y, batch):
        x = to_dense_batch(x, batch)[0]
        y = to_dense_batch(y, batch)[0] 

        # https://github.com/zichunhao/mnist_graph_autoencoder/blob/master/utils/loss.py
        dist = pairwise_distance(x, y, self.device)

        min_dist_xy = torch.min(dist, dim=-1)
        min_dist_yx = torch.min(dist, dim=-2)  # Equivalent to permute the last two axis

        loss = torch.sum(min_dist_xy.values + min_dist_yx.values) / len(x)

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
        # px py pz -> pt eta phi
        data = preprocess_emdnn_input(x, y, batch)
        out = self.emd_model(data)
        emd = out[0]
        return emd.mean()

    def deepemd_loss(self, x, y, batch, l2_strength=1e-4):
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        # pt eta phi -> eta phi pt
        inds = torch.LongTensor([1,2,0]).to(self.device)
        x = torch.index_select(x, 1, inds)
        y = torch.index_select(y, 1, inds)
        # format shape as [nbatch, nparticles(padded), features]
        x = to_dense_batch(x,batch)[0]
        y = to_dense_batch(y,batch)[0]
        # get loss using raghav's implementation of DeepEmd
        emd = deepemd(x, y, device=self.device, l2_strength=l2_strength)
        return emd

    def mse(self):
        pass

    def emd_in_forward(self):
        pass
