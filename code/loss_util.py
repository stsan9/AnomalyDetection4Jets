import sys
import torch
import emd_models
import numpy as np
import torch_scatter
import os.path as osp
from torch_geometric.data import Data

multi_gpu = torch.cuda.device_count()>1

def load_emd_model(modname, device):
    emd_model = getattr(emd_models, modname[:-9])(device=device)
    modpath = osp.join('/anomalyvol/emd_models/', modname)
    emd_model.load_state_dict(torch.load(modpath, map_location=torch.device(device)))
    return emd_model

def arctanh(x):
    return torch.log1p(2*x/(1-x)) / 2

def get_ptetaphi(x,batch):
    px = x[:,0]
    py = x[:,1]
    pz = x[:,2]
    p = torch.sqrt(torch.square(px) + torch.square(py) + torch.square(pz))
    pt = torch.sqrt(torch.square(px) + torch.square(py))
    eta = arctanh(pz / (p + 1e-12))
    phi = torch.atan(py / (px + 1e-12))
    ts = [px,py,pz,p,pt,eta,phi]
    for e in ts:
        if True in torch.isnan(e):
            raise ValueError('nan in get_ptetaphi')
    mat = torch.stack((pt,eta,phi),dim=1)
    # center by pt centroid while accounting for torch geo batching
    n = torch_scatter.scatter(mat[:,1:3].clone() * mat[:,0,None].clone(), batch, dim=0, reduce='sum')
    d = torch_scatter.scatter(mat[:,0], batch, dim=0, reduce='sum')
    yphi_avg = (n.T / d).T  # returns yphi_avg for each batch
    _, counts = torch.unique_consecutive(batch, return_counts=True)
    yphi_avg = torch.repeat_interleave(yphi_avg, counts, dim=0) # repeat per batch for subtraction step
    mat[:,1:3] -= yphi_avg
    return mat

class LossFunction:
    def __init__(self, lossname, emd_modname='EmdNNRel.best.pth', device='cuda:0'):
        if lossname == 'mse':
            loss = torch.nn.MSELoss(reduction='mean')
        else:
            loss = getattr(self, lossname)
            if lossname == 'emd_loss' and not multi_gpu:
                # keep emd model in memory
                # if using DataParallel it's merged into the network's forward pass to distribute gpu memory
                self.emd_model = load_emd_model(emd_modname,device)
        self.name = lossname
        self.loss_ftn = loss
        self.device = device

    def chamfer_loss(self, x,y):
        nparts = x.shape[0]
        dist = torch.pow(torch.cdist(x,y),2)
        in_dist_out = torch.min(dist,dim=0)
        out_dist_in = torch.min(dist,dim=1)
        loss = torch.sum(in_dist_out.values + out_dist_in.values) / nparts
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
        Ex = torch_scatter.scatter(src=x[:,0],index=batch)
        Ey = torch_scatter.scatter(src=y[:,0],index=batch)
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        Ex_repeat = torch.repeat_interleave(Ex, counts, dim=0)
        Ey_repeat = torch.repeat_interleave(Ey, counts, dim=0)
        x[:,0] = x[:,0].clone() / Ex_repeat
        y[:,0] = y[:,0].clone() / Ey_repeat
        # create data object for emd model
        jet_pair = torch.cat((x,y),0)
        u = torch.cat((Ex.view(-1,1),Ey.view(-1,1)),dim=1) / 100.0
        data = Data(x=jet_pair, batch=torch.cat((batch,batch)), u=u).to(self.device)
        # get emd between x and y
        out = self.emd_model(data)
        emd = out[0]    # ignore other model outputs
        return emd
