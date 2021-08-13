import torch
from torch_geometric.data import Batch

import models.models as models

def get_model(modname, **kwargs):
    input_dim = kwargs['input_dim']
    big_dim = kwargs['big_dim']
    hidden_dim = kwargs['hidden_dim']
    emd_modname = kwargs['emd_modname']

    if modname == 'MetaLayerGAE':
        model = models.GNNAutoEncoder()
    else:
        if modname[-3:] == 'EMD':
            model = getattr(models, modname)(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim, emd_modname=emd_modname)
        else:
            model = getattr(models, modname)(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim)
    return model

# helper to perform correct loss
def forward_loss(model, data, loss_ftn_obj, device, multi_gpu):
    
    if not multi_gpu:
        data = data.to(device)

    if 'emd_loss' in loss_ftn_obj.name or loss_ftn_obj.name == 'chamfer_loss':
        batch_output = model(data)
        if multi_gpu:
            data = Batch.from_data_list(data).to(device)
        y = data.x
        batch = data.batch
        batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, batch)

    elif loss_ftn_obj.name == 'emd_in_forward':
        _, batch_loss = model(data)
        batch_loss = batch_loss.mean()

    elif loss_ftn_obj.name == 'vae_loss':
        batch_output, mu, log_var = model(data)
        y = torch.cat([d.x for d in data]).to(device) if multi_gpu else data.x
        y = y.contiguous()
        batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, mu, log_var)

    else:
        batch_output = model(data)
        y = torch.cat([d.x for d in data]).to(device) if multi_gpu else data.x
        y = y.contiguous()
        batch_loss = loss_ftn_obj.loss_ftn(batch_output, y)
    return batch_loss