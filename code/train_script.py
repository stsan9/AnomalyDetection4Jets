import glob
import math
import tqdm
import torch
import random
import torch.nn as nn
import os.path as osp
from torch.utils.data import random_split
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch

import models
import emd_models
from loss_util import LossFunction
from graph_data import GraphDataset

torch.manual_seed(0)

# train and test helper functions
@torch.no_grad()
def test(model, loader, total, batch_size, loss_ftn_obj, no_E = False):
    model.eval()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        if data.x.shape[0] <= 1:    # skip strange jets
            continue
        data = data.to(device)

        # format data
        if (loss_ftn_obj.name == "emd_loss"):
            data.x = data.x[:,4:-1] # pt, eta, phi
        elif (no_E == True):
            data.x = data.x[:,:3]   # px, py, pz
        y = data.x
        y = y.contiguous()

        # forward and loss
        if loss_ftn_obj.name == "vae_loss":
            batch_output, mu, log_var = model(data)
            batch_loss_item = loss_ftn_obj.loss_ftn(batch_output, y, mu, log_var).item()
        elif loss_ftn_obj.name == "emd_loss":
            batch_output = model(data)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, data.batch)
            # square (for positivity) and avg into one val
            batch_loss_item = batch_loss.mean().item()
        else:
            batch_output = model(data)
            batch_loss_item = loss_ftn_obj.loss_ftn(batch_output, y).item()
        sum_loss += batch_loss_item
        t.set_description("loss = %.5f" % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def train(model, optimizer, loader, total, batch_size, loss_ftn_obj, no_E = False):
    model.train()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        if data.x.shape[0] <= 1:    # skip strange jets
            continue
        data = data.to(device)

        # format data
        if (loss_ftn_obj.name == "emd_loss"):
            data.x = data.x[:,4:-1]
        elif (no_E == True):
            data.x = data.x[:,:3]
        y = data.x
        y = y.contiguous()
        optimizer.zero_grad()

        # forward pass and loss calc
        if loss_ftn_obj.name == "vae_loss":
            batch_output, mu, log_var = model(data)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, mu, log_var)
        elif loss_ftn_obj.name == "emd_loss":
            batch_output = model(data)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, data.batch - torch.min(data.batch))
            batch_loss = batch_loss.mean()
        else:
            batch_output = model(data)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y)

        # update
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description("loss = %.5f" % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()

    return sum_loss/(i+1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod-name", type=str, help="model name for saving and loading", required=True)
    parser.add_argument("--input-dir", type=str, help="location of dataset", required=True)
    parser.add_argument("--box-num", type=int, help="0=QCD-background; 1=bb1; 2=bb2; 4=rnd", default=0, required=False)
    parser.add_argument("--lat-dim", type=int, help="latent space size", default=2, required=False)
    parser.add_argument("--no-E", action='store_true', 
                        help="toggle to remove energy from training and testing", default=False, required=False)
    parser.add_argument("--model", choices=models.model_list, help="model selection", required=True)
    parser.add_argument("--batch-size", type=int, help="batch size", default=2, required=False)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3, required=False)
    parser.add_argument("--loss", choices=["chamfer_loss","emd_loss","vae_loss","mse"], help="loss function", required=True)
    parser.add_argument("--emd-model-name", choices=[osp.basename(x) for x in glob.glob('/anomalyvol/emd_models/*')], 
                        help="emd models for loss", default='Symmetric1k.best.pth', required=False)
    args = parser.parse_args()
    batch_size = args.batch_size

    # get dataset and split
    gdata = GraphDataset(root=args.input_dir, bb=args.box_num)
    # merge data from separate files into one contiguous array
    bag = []
    for g in gdata:
        bag += g
    random.Random(0).shuffle(bag)
    # 80:10:10 split datasets
    fulllen = len(bag)
    train_len = int(0.8 * fulllen)
    tv_len = int(0.10 * fulllen)
    train_dataset = bag[:train_len]
    valid_dataset = bag[train_len:train_len + tv_len]
    test_dataset  = bag[train_len + tv_len:]
    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)
    test_samples = len(test_dataset)
    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    # create model
    no_E = args.no_E
    input_dim = 3 if (args.no_E or args.loss=='emd_loss') else 4
    big_dim = 32
    hidden_dim = args.lat_dim
    fulllen = len(gdata)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    n_epochs = 200
    lr = args.lr
    patience = 10
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_fname = args.mod_name
    if args.model == 'MetaLayerGAE':
        model = models.GNNAutoEncoder().to(device)
    else:
        model = getattr(models, args.model)(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    # load in model
    modpath = osp.join('/anomalyvol/models/',model_fname+'.best.pth')
    try:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(modpath, map_location=torch.device('cuda')))
        else:
            model.load_state_dict(torch.load(modpath, map_location=torch.device('cpu')))
        print("Loaded model")
    except:
        print("Creating new model")

    # specify loss function
    loss_ftn_obj = LossFunction(args.loss, emd_modname=args.emd_model_name, device=device)

    # Training loop
    stale_epochs = 0
    best_valid_loss = 9999999
    loss = best_valid_loss
    best_valid_loss = test(model, valid_loader, valid_samples, batch_size, loss_ftn_obj, no_E)

    for epoch in range(0, n_epochs):
        loss = train(model, optimizer, train_loader, train_samples, batch_size, loss_ftn_obj, no_E)
        valid_loss = test(model, valid_loader, valid_samples, batch_size, loss_ftn_obj, no_E)
        print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
        print('               Validation Loss: {:.4f}'.format(valid_loss))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            modpath = osp.join('/anomalyvol/models/',model_fname+'.best.pth')
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
