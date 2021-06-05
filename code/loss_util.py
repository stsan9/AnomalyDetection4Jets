import torch
import torch_scatter
import os.path as osp
import emd_models
import sys
from torch_geometric.data import Data
from emd_loss import emd_loss as deepemd
from torch_geometric.utils import to_dense_batch

def arctanh(x):
    return torch.log1p(2*x/(1-x)) / 2

def get_ptetaphi(x):
    px = x[:,0]
    py = x[:,1]
    pz = x[:,2]
    p = torch.sqrt(torch.square(px) + torch.square(py) + torch.square(pz))
    pt = torch.sqrt(torch.square(px) + torch.square(py))
    eta = arctanh(pz / (p + 10e-12))
    phi = torch.atan(py / (px + 10e-12))
    if True in torch.isnan(eta):
        print("NAN in eta")
    if True in torch.isnan(phi):
        print("NAN in phi")
    if True in torch.isnan(pz / p):
        print("NAN in pz/p")
    if True in torch.isnan(py / px):
        print("NAN in py/px")
    mat = torch.stack((pt,eta,phi),dim=1)
    return mat

class LossFunction:
    def __init__(self, lossname, emd_modname="Symmetric1k.best.pth", device='cuda:0'):
        if lossname == 'mse':
            loss = torch.nn.MSELoss(reduction='mean')
        elif lossname == 'deep_emd_loss':   # use adaptation by raghav of deepemd paper
            loss = self.deep_emd_loss
        else:
            loss = getattr(self, lossname)
            if lossname == 'emd_loss':  # use neural network
                self.emd_model = self.load_emd_model(emd_modname,device)
        self.name = lossname
        self.loss_ftn = loss
        self.device = device

    def load_emd_model(self, modname, device):
        if modname == 'emd_rel_1k.best.pth':
            emd_model = emd_models.SymmetricDDEdgeNetRel(device=device)
        elif modname == 'emd_spl_1k.best.pth':
            emd_model = emd_models.SymmetricDDEdgeNetSpl(device=device)
        else:
            emd_model = emd_models.SymmetricDDEdgeNet(device=device)
        modpath = osp.join("/anomalyvol/emd_models/", modname)
        emd_model.load_state_dict(torch.load(modpath, map_location=torch.device(device)))
        return emd_model

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
        x = get_ptetaphi(x)
        y = get_ptetaphi(y)
        # concatenate column of 1s to one jet and -1 to other jet
        x = torch.cat((x,torch.ones(len(x),1).to(device)), 1)
        y = torch.cat((y,torch.ones(len(y),1).to(device)*-1), 1)
        jet_pair = torch.cat((x,y),0)
        # create data object for emd model
        Ei = torch_scatter.scatter(src=x[:,0],index=batch)
        Ey = torch_scatter.scatter(src=y[:,0],index=batch)
        u = torch.cat((Ei.view(-1,1),Ey.view(-1,1)),dim=1) / 100.0
        data = Data(x=jet_pair, batch=torch.cat((batch,batch)), u=u).to(self.device)
        # get emd between x and y
        out = self.emd_model(data)
        emd = out[0]    # ignore other model outputs
        return emd

    def deep_emd_loss(self, x, y, batch, l2_strength=1e-4):
        x = get_ptetaphi(x)
        y = get_ptetaphi(y)
        # eta phi pt
        inds = torch.LongTensor([1,2,0]).to(self.device)
        x = torch.index_select(x, 1, inds)
        y = torch.index_select(y, 1, inds)
        # format shape as [nbatch, nparticles(padded), features]
        x = to_dense_batch(x,batch)[0]
        y = to_dense_batch(y,batch)[0]
        # get loss using raghav's implementation of DeepEmd
        emd = deepemd(x, y, device=self.device, l2_strength=l2_strength)
        return emd
