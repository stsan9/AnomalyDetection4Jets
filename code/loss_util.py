import torch
import torch_scatter
import os.path as osp
import emd_models
import sys
from torch_geometric.data import Data

class LossFunction:
    def __init__(self, lossname, emd_modname="Symmetric1k.best.pth", device='cuda:0'):
        if lossname == 'mse':
            loss = torch.nn.MSELoss(reduction='mean')
        else:
            loss = getattr(self, lossname)
            if lossname == 'emd_loss':
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
