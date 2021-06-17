import glob
import math
import tqdm
import torch
import random
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
from plot_util import loss_curves

torch.manual_seed(0)
multi_gpu = torch.cuda.device_count()>1

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
        if (no_E == True):
            data.x = data.x[:,:3]   # px, py, pz
        y = data.x
        y = y.contiguous()

        # forward and loss
        if loss_ftn_obj.name == 'vae_loss':
            batch_output, mu, log_var = model(data)
            batch_loss_item = loss_ftn_obj.loss_ftn(batch_output, y, mu, log_var).item()
        elif loss_ftn_obj.name == 'emd_loss':
            batch_output = model(data)
            try:
                batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, data.batch)
            except RuntimeError as e:
                torch.save([data],'/anomalyvol/debug/debug_input.pt')
                torch.save(model.state_dict(),'/anomalyvol/debug/debug_model.pth')
                raise RuntimeError('found nan in loss') from e
                # exit('Check debug directory for model and input')
            # square (for positivity) and avg into one val
            batch_loss_item = batch_loss.mean().item()
        else:
            batch_output = model(data)
            batch_loss_item = loss_ftn_obj.loss_ftn(batch_output, y).item()
        sum_loss += batch_loss_item
        t.set_description('eval loss = %.5f' % (batch_loss_item))
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
        if (no_E == True):
            data.x = data.x[:,:3]
        y = data.x
        y = y.contiguous()
        optimizer.zero_grad()

        # forward pass and loss calc
        if loss_ftn_obj.name == 'vae_loss':
            batch_output, mu, log_var = model(data)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, mu, log_var)
        elif loss_ftn_obj.name == 'emd_loss':
            batch_output = model(data)
            try:
                batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, data.batch)
            except ValueError as e:
                torch.save([data],'/anomalyvol/debug/debug_input.pt')
                torch.save(model.state_dict(),'/anomalyvol/debug/debug_model.pth')
                exit('Check debug directory for model and input')
            batch_loss = batch_loss.mean()
        else:
            batch_output = model(data)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y)

        # update
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description('train loss = %.5f' % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()

    return sum_loss / (i+1)

# train and test helper functions
@torch.no_grad()
def test_parallel(model, loader, total, batch_size, loss_ftn_obj, no_E = False):
    model.eval()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:

        # forward and loss
        if loss_ftn_obj.name == 'vae_loss':
            batch_output, mu, log_var = model(data)
            y = torch.cat([d.x for d in data]).to(device)
            batch_loss_item = loss_ftn_obj.loss_ftn(batch_output, y, mu, log_var).item()
        elif loss_ftn_obj.name == 'emd_loss':
            batch_output = model(data)
            data_batch = Batch.from_data_list(data).to(device)
            try:
                batch_loss = loss_ftn_obj.loss_ftn(batch_output, data_batch.x, data_batch.batch)
            except RuntimeError as e:
                torch.save([data],'/anomalyvol/debug/debug_input.pt')
                torch.save(model.state_dict(),'/anomalyvol/debug/debug_model.pth')
                raise RuntimeError('found nan in loss') from e
                # exit('Check debug directory for model and input')
            # square (for positivity) and avg into one val
            batch_loss_item = batch_loss.mean().item()
        else:
            batch_output = model(data)
            y = torch.cat([d.x for d in data]).to(device)
            batch_loss_item = loss_ftn_obj.loss_ftn(batch_output, y).item()

        sum_loss += batch_loss_item
        t.set_description('eval loss = %.5f' % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def train_parallel(model, optimizer, loader, total, batch_size, loss_ftn_obj, no_E = False):
    model.train()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        optimizer.zero_grad()

        # forward pass and loss calc
        if loss_ftn_obj.name == 'vae_loss':
            batch_output, mu, log_var = model(data)
            y = torch.cat([d.x for d in data]).to(device)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, mu, log_var)
        elif loss_ftn_obj.name == 'emd_loss':
            batch_output = model(data)
            data_batch = Batch.from_data_list(data).to(device)
            try:
                batch_loss = loss_ftn_obj.loss_ftn(batch_output, data_batch.x, data_batch.batch)
            except ValueError as e:
                torch.save([data],'/anomalyvol/debug/debug_input.pt')
                torch.save(model.state_dict(),'/anomalyvol/debug/debug_model.pth')
                exit('Check debug directory for model and input')
            batch_loss = batch_loss.mean()
        else:
            batch_output = model(data)
            y = torch.cat([d.x for d in data]).to(device)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, y)

        # update
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description('train loss = %.5f' % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()

    return sum_loss / (i+1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod-name', type=str, help='model name for saving and loading', required=True)
    parser.add_argument('--input-dir', type=str, help='location of dataset', required=True)
    parser.add_argument('--output-dir', type=str, help='root folder to output experiment results to', default='/anomalyvol/experiments/', required=False)
    parser.add_argument('--box-num', type=int, help='0=QCD-background; 1=bb1; 2=bb2; 4=rnd', default=0, required=False)
    parser.add_argument('--lat-dim', type=int, help='latent space size', default=2, required=False)
    parser.add_argument('--no-E', action='store_true', 
                        help='toggle to remove energy from training and testing', default=True, required=False)
    parser.add_argument('--model', choices=models.model_list, help='model selection', required=True)
    parser.add_argument('--batch-size', type=int, help='batch size', default=2, required=False)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3, required=False)
    parser.add_argument('--patience', type=int, help='patience', default=10, required=False)
    parser.add_argument('--loss', choices=['chamfer_loss','emd_loss','vae_loss','mse'], help='loss function', required=True)
    parser.add_argument('--emd-model-name', choices=[osp.basename(x) for x in glob.glob('/anomalyvol/emd_models/*')], 
                        help='emd models for loss', default='Symmetric1k.best.pth', required=False)
    parser.add_argument('--num-data', type=int, help='how much data to use (e.g. 10 jets)', default=None, required=False)
    args = parser.parse_args()
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
    # 80:10:10 split datasets
    fulllen = len(bag)
    train_len = int(0.8 * fulllen)
    train_dataset = bag[:train_len]
    valid_dataset = bag[train_len:]
    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)
    # dataloaders
    if multi_gpu:
        train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    # specify loss function
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    loss_ftn_obj = LossFunction(args.loss, emd_modname=args.emd_model_name, device=device)

    # create model
    no_E = args.no_E
    input_dim = 3 if (args.no_E or args.loss=='emd_loss') else 4
    big_dim = 32
    hidden_dim = args.lat_dim
    n_epochs = 200
    lr = args.lr
    patience = args.patience
    if args.model == 'MetaLayerGAE':
        model = models.GNNAutoEncoder()
    else:
        model = getattr(models, args.model)(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    # load in model
    modpath = osp.join(save_dir,model_fname+'.best.pth')
    try:
        model.load_state_dict(torch.load(modpath))
        print('Loaded model')
        best_valid_loss = test(model, valid_loader, valid_samples, batch_size, loss_ftn_obj, no_E)
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

    valid_losses = []
    train_losses = []
    for epoch in range(0, n_epochs):
        try:
            if multi_gpu:
                loss = train_parallel(model, optimizer, train_loader, train_samples, batch_size, loss_ftn_obj, no_E)
                valid_loss = test_parallel(model, valid_loader, valid_samples, batch_size, loss_ftn_obj, no_E)
            else:
                loss = train(model, optimizer, train_loader, train_samples, batch_size, loss_ftn_obj, no_E)
                valid_loss = test(model, valid_loader, valid_samples, batch_size, loss_ftn_obj, no_E)
            scheduler.step(valid_loss)
            train_losses.append(loss)
            valid_losses.append(valid_loss)
        except RuntimeError as e:   # catch errors during runtime to save loss curves
            if epoch > 3:
                train_epochs = list(range(epoch+1))
                early_stop_epoch = epoch - stale_epochs
                # adjust list lengths by whichever broke before plotting
                valid_epochs = min(len(train_epochs), len(train_losses), len(valid_losses))
                early_stop_epoch -= len(train_epochs) - valid_epochs
                train_epochs = train_epochs[:valid_epochs]
                train_losses = train_losses[:valid_epochs]
                valid_losses = valid_losses[:valid_epochs]
                loss_curves(train_epochs, early_stop_epoch, train_losses, valid_losses, save_dir)
            print('Error during training',e)
            exit('Exiting Early')

        print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
        print('               Validation Loss: {:.4f}'.format(valid_loss))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('New best model saved to:',modpath)
            torch.save(model.state_dict(),modpath)
            stale_epochs = 0
        else:
            print(f'Stale epoch\nBest: {best_valid_loss}\nCurr: {valid_loss}')
            stale_epochs += 1
        if stale_epochs >= patience:
            print('Early stopping after %i stale epochs'%patience)
            break
    train_epochs = list(range(epoch+1))
    early_stop_epoch = epoch - stale_epochs + 1
    loss_curves(train_epochs, early_stop_epoch, train_losses, valid_losses, save_dir)
            
    print('Completed')
