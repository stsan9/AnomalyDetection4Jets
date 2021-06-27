import glob
import math
import tqdm
import torch
import random
import inspect
import torch.nn as nn
import os.path as osp
from pathlib import Path
from torch.utils.data import random_split
from torch_geometric.nn import EdgeConv, global_mean_pool, DataParallel
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch

import models
import emd_models
from loss_util import LossFunction
from graph_data import GraphDataset
from plot_util import loss_curves, plot_reco_difference

torch.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
multi_gpu = torch.cuda.device_count()>1

@torch.no_grad()
def test(model, loader, total, batch_size, loss_ftn_obj):
    model.eval()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        if data.x.shape[0] <= 1:    # skip strange jets
            continue
        data = data.to(device)

        y = data.x
        y = y.contiguous()

        # forward and loss
        if loss_ftn_obj.name == 'vae_loss':
            batch_output, mu, log_var = model(data)
            batch_loss_item = loss_ftn_obj.loss_ftn(batch_output, y, mu, log_var).item()
        elif loss_ftn_obj.name == 'emd_loss' or loss_ftn_obj.name == 'chamfer_loss':
            batch_output = model(data)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, data.batch)
            batch_loss_item = batch_loss.mean().item()
        else:
            batch_output = model(data)
            batch_loss_item = loss_ftn_obj.loss_ftn(batch_output, y).item()
        sum_loss += batch_loss_item
        t.set_description('eval loss = %.5f' % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def train(model, optimizer, loader, total, batch_size, loss_ftn_obj):
    model.train()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        if data.x.shape[0] <= 1:    # skip strange jets
            continue
        data = data.to(device)

        y = data.x
        y = y.contiguous()
        optimizer.zero_grad()

        # forward pass and loss calc
        if loss_ftn_obj.name == 'vae_loss':
            batch_output, mu, log_var = model(data)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, mu, log_var)
        elif loss_ftn_obj.name == 'emd_loss' or loss_ftn_obj.name == 'chamfer_loss':
            batch_output = model(data)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, data.batch)
            batch_loss = batch_loss.mean()
        else:
            batch_output = model(data)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y)

        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description('train loss = %.5f' % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()

    return sum_loss / (i+1)

@torch.no_grad()
def test_parallel(model, loader, total, batch_size, loss_ftn_obj):
    model.eval()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:

        # forward and loss
        if loss_ftn_obj.name == 'vae_loss':
            batch_output, mu, log_var = model(data)
            y = torch.cat([d.x for d in data]).to(device)
            batch_loss_item = loss_ftn_obj.loss_ftn(batch_output, y, mu, log_var).item()
        elif loss_ftn_obj.name == 'emd_loss' or loss_ftn_obj.name == 'chamfer_loss':
            batch_output = model(data)
            data_batch = Batch.from_data_list(data).to(device)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, data_batch.x, data_batch.batch)
            batch_loss_item = batch_loss.mean().item()
        elif loss_ftn_obj.name == 'emd_loss_layer':
            _, batch_loss = model(data)
            batch_loss_item = batch_loss.mean().item()
        else:
            batch_output = model(data)
            y = torch.cat([d.x for d in data]).to(device)
            batch_loss_item = loss_ftn_obj.loss_ftn(batch_output, y).item()

        sum_loss += batch_loss_item
        t.set_description('eval loss = %.5f' % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def train_parallel(model, optimizer, loader, total, batch_size, loss_ftn_obj):
    model.train()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        optimizer.zero_grad()

        if loss_ftn_obj.name == 'vae_loss':
            batch_output, mu, log_var = model(data)
            y = torch.cat([d.x for d in data]).to(device)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, mu, log_var)
        elif loss_ftn_obj.name == 'emd_loss' or loss_ftn_obj.name == 'chamfer_loss':
            batch_output = model(data)
            data_batch = Batch.from_data_list(data).to(device)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, data_batch.x, data_batch.batch)
            batch_loss = batch_loss.mean()
        elif loss_ftn_obj.name == 'emd_loss_layer':
            _, batch_loss = model(data)
            batch_loss = batch_loss.mean()
        else:
            batch_output = model(data)
            y = torch.cat([d.x for d in data]).to(device)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y)

        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description('train loss = %.5f' % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()

    return sum_loss / (i+1)

def main(args):
    batch_size = args.batch_size
    model_fname = args.mod_name

    if multi_gpu and batch_size < torch.cuda.device_count():
        exit('Batch size too small')

    # make a folder for the graphs of this model
    Path(args.output_dir).mkdir(exist_ok=True)
    save_dir = osp.join(args.output_dir,model_fname)
    Path(save_dir).mkdir(exist_ok=True)

    # get dataset and split
    gdata = GraphDataset(root=args.input_dir, bb=args.box_num)
    # merge data from separate files into one contiguous array
    bag = []
    for g in gdata:
        bag += g
    random.Random(0).shuffle(bag)
    bag = bag[:args.num_data]
    # temporary patch to use px, py, pz
    if args.swap_input:
        for d in bag:   # pt eta phi
            d.x = d.x[:,5:8]
    else:
        for d in bag:   # px py pz
            d.x = d.x[:,:3]
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
    if multi_gpu:
        train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
        test_loader = DataListLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    # specify loss function
    loss_ftn_obj = LossFunction(args.loss, emd_modname=args.emd_model_name, device=device, swap_input=args.swap_input)

    # create model
    input_dim = 3
    big_dim = 32
    hidden_dim = args.lat_dim
    lr = args.lr
    patience = args.patience

    if args.model == 'MetaLayerGAE':
        model = models.GNNAutoEncoder()
    else:
        if args.model[-3:] == 'EMD':
            model = getattr(models, args.model)(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim, emd_modname=args.emd_model_name)
        else:
            model = getattr(models, args.model)(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)

    valid_losses = []
    train_losses = []
    start_epoch = 0
    n_epochs = 200

    # load in model
    modpath = osp.join(save_dir,model_fname+'.best.pth')
    try:
        model.load_state_dict(torch.load(modpath))
        train_losses, valid_losses, start_epoch = torch.load(osp.join(save_dir,'losses.pt'))
        print('Loaded model')
        best_valid_loss = test(model, valid_loader, valid_samples, batch_size, loss_ftn_obj)
        print(f'Saved model valid loss: {best_valid_loss}')
    except:
        print('Creating new model')
        best_valid_loss = 9999999
    if multi_gpu:
        model = DataParallel(model)
    model.to(torch.device(device))

    # Training loop
    stale_epochs = 0
    loss = best_valid_loss
    for epoch in range(start_epoch, n_epochs):

        if multi_gpu:
            loss = train_parallel(model, optimizer, train_loader, train_samples, batch_size, loss_ftn_obj)
            valid_loss = test_parallel(model, valid_loader, valid_samples, batch_size, loss_ftn_obj)
        else:
            loss = train(model, optimizer, train_loader, train_samples, batch_size, loss_ftn_obj)
            valid_loss = test(model, valid_loader, valid_samples, batch_size, loss_ftn_obj)

        scheduler.step(valid_loss)
        train_losses.append(loss)
        valid_losses.append(valid_loss)
        print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
        print('               Validation Loss: {:.4f}'.format(valid_loss))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('New best model saved to:',modpath)
            if multi_gpu:
                torch.save(model.module.state_dict(), modpath)
            else:
                torch.save(model.state_dict(), modpath)
            torch.save((train_losses, valid_losses, epoch+1), osp.join(save_dir,'losses.pt'))
            stale_epochs = 0
        else:
            stale_epochs += 1
            print(f'Stale epoch: {stale_epochs}\nBest: {best_valid_loss}\nCurr: {valid_loss}')
        if stale_epochs >= patience:
            print('Early stopping after %i stale epochs'%patience)
            break

    # model training done
    train_epochs = list(range(epoch+1))
    early_stop_epoch = epoch - stale_epochs
    loss_curves(train_epochs, early_stop_epoch, train_losses, valid_losses, save_dir)

    # compare input and reconstructions
    model.load_state_dict(torch.load(modpath))
    input_fts = []
    reco_fts = []
    for t in valid_loader:
        model.eval()
        if isinstance(t, list):
            for d in t:
                input_fts.append(d.x)
        else:
            input_fts.append(t.x)
            t.to(device)
        reco_out = model(t)
        if isinstance(reco_out, tuple):
            reco_out = reco_out[0]
        reco_fts.append(reco_out.cpu().detach())
    input_fts = torch.cat(input_fts)
    reco_fts = torch.cat(reco_fts)
    plot_reco_difference(input_fts, reco_fts, model_fname, osp.join(save_dir, 'reconstruction_post_train', 'valid'))

    input_fts = []
    reco_fts = []
    for t in test_loader:
        model.eval()
        if isinstance(t, list):
            for d in t:
                input_fts.append(d.x)
        else:
            input_fts.append(t.x)
            t.to(device)
        reco_out = model(t)
        if isinstance(reco_out, tuple):
            reco_out = reco_out[0]
        reco_fts.append(reco_out.cpu().detach())
    input_fts = torch.cat(input_fts)
    reco_fts = torch.cat(reco_fts)
    plot_reco_difference(input_fts, reco_fts, model_fname, osp.join(save_dir, 'reconstruction_post_train', 'test'))
    print('Completed')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod-name', type=str, help='model name for saving and loading', required=True)
    parser.add_argument('--input-dir', type=str, help='location of dataset', required=True)
    parser.add_argument('--output-dir', type=str, help='root folder to output experiment results to', 
                        default='/anomalyvol/experiments/', required=False)
    parser.add_argument('--box-num', type=int, help='0=QCD-background; 1=bb1; 2=bb2; 4=rnd', default=0, required=False)
    parser.add_argument('--lat-dim', type=int, help='latent space size', default=2, required=False)
    parser.add_argument('--model', 
                        choices=[m[0] for m in inspect.getmembers(models, inspect.isclass) if m[1].__module__ == 'models'], 
                        help='model selection', required=True)
    parser.add_argument('--batch-size', type=int, help='batch size', default=2, required=False)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3, required=False)
    parser.add_argument('--patience', type=int, help='patience', default=10, required=False)
    parser.add_argument('--loss', choices=['chamfer_loss','emd_loss','vae_loss','mse','emd_loss_layer'], 
                        help='loss function', required=True)
    parser.add_argument('--emd-model-name', choices=[osp.basename(x) for x in glob.glob('/anomalyvol/emd_models/*')], 
                        help='emd models for loss', default='Symmetric1k.best.pth', required=False)
    parser.add_argument('--num-data', type=int, help='how much data to use (e.g. 10 jets)', 
                        default=None, required=False)
    parser.add_argument("--swap-input", action="store_true", default=False, help="use pt eta phi instead of 3 momentum")
    args = parser.parse_args()

    main(args)
