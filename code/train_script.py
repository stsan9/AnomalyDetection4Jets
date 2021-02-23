import glob
import torch
import tqdm
import math
import torch.nn as nn
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from torch.utils.data import random_split
import os.path as osp
from graph_data import GraphDataset, collate
import models
import loss_util

torch.manual_seed(0)

# train and test helper functions
@torch.no_grad()
def test(model, loader, total, batch_size, loss_ftn, no_E = False):
    model.eval()
    
    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        if data.x.shape[0] <= 1:    # skip strange jets
            continue
        data = data.to(device)
        if (loss_ftn == loss_util.emd):
            data.x = data.x[:,4:-1]
        elif (no_E == True):
            data.x = data.x[:,:3]
        y = data.x # the model will overwrite data.x, so save a copy
        y = y.contiguous()
        if loss_ftn == loss_util.vae_loss:
            batch_output, mu, log_var = model(data)
            batch_loss_item = loss_ftn(batch_output, y, mu, log_var).item()
        else:
            batch_output = model(data)
            batch_loss_item = loss_ftn(batch_output, y).item()
        sum_loss += batch_loss_item
        t.set_description("loss = %.5f" % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def train(model, optimizer, loader, total, batch_size, loss_ftn, no_E = False):
    model.train()
    
    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        if data.x.shape[0] <= 1:    # skip strange jets
            continue
        data = data.to(device)
        if (loss_ftn == loss_util.emd):
            data.x = data.x[:,4:-1]
        elif (no_E == True):
            data.x = data.x[:,:3]
        y = data.x # the model will overwrite data.x, so save a copy
        y = y.contiguous()
        optimizer.zero_grad()
        if loss_ftn == loss_util.vae_loss:
            batch_output, mu, log_var = model(data)
            batch_loss = loss_ftn(batch_output, y, mu, log_var)
        else:
            batch_output = model(data)
            batch_loss = loss_ftn(batch_output, y)
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description("loss = %.5f" % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()
    
    return sum_loss/(i+1)

if __name__ == "__main__":
    # display some argument options
    saved_models = [osp.basename(x)[:-9] for x in glob.glob('/anomalyvol/models/*')]
    print(f"saved mod_name's:\n{saved_models}\n")
    print(f"model_num options:\n{models.model_list}\n")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod_name", type=str, help="model name for saving and loading", required=True)
    parser.add_argument("--input-dir", type=str, help="location of dataset", required=True)
    parser.add_argument("--box_num", type=int, help="0=QCD-background; 1=bb1; 2=bb2; 4=rnd", default=0, required=False)
    parser.add_argument("--lat_dim", type=int, help="latent space size", default=2, required=False)
    parser.add_argument("--no_E", action='store_true', 
                        help="toggle to remove energy from training and testing", default=False, required=False)
    parser.add_argument("--metalayer", action='store_true', 
                        help="toggle to use metalayer model", default=False, required=False)
    parser.add_argument("--model_num", type=int, help="model number", default=-1, required=True)
    parser.add_argument("--batch_size", type=int, help="batch size", default=2, required=False)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3, required=False)
    parser.add_argument("--loss", choices=["chamfer_loss","emd","vae_loss","mse"], help="loss function" default="mse")
    args = parser.parse_args()
    batch_size = args.batch_size

    # specify loss function
    if args.loss != mse:
        loss_ftn = getattr(loss_util, args.loss)
    else:
        loss_ftn = nn.MSELoss(reduction='mean')

    # get dataset and split
    gdata = GraphDataset(root=osp.join(args.input_dir, bb=args.box_num)
    train_dataset, valid_dataset, test_dataset = random_split(gdata, [fulllen-2*tv_num,tv_num,tv_num])
    if args.loss == "mse":  # collate dataset
        train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        train_loader.collate_fn = collate
        valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
        valid_loader.collate_fn = collate
        test_loader = DataListLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
        test_loader.collate_fn = collate
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)
    test_samples = len(test_dataset)

    # create model
    no_E = args.no_E
    input_dim = 3 if args.no_E else 4
    big_dim = 32
    hidden_dim = args.lat_dim
    fulllen = len(gdata)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    n_epochs = 200
    lr = args.lr
    patience = 10
    device = 'cuda:0'
    model_fname = args.mod_name
    if args.metalayer:
        print("Using metalayer model")
        model = models.GNNAutoEncoder().to(device)
    elif args.model_num != -1:
        model = models.model_list[args.model_num](input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim).to(device)
    else:
        print("Using default EdgeConv")
        model = models.EdgeNet(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim).to(device)
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

    # Training loop
    stale_epochs = 0
    best_valid_loss = 9999999
    loss = best_valid_loss
    best_valid_loss = test(model, valid_loader, valid_samples, batch_size, loss_ftn, no_E)

    for epoch in range(0, n_epochs):
        loss = train(model, optimizer, train_loader, train_samples, batch_size, loss_ftn, no_E)
        valid_loss = test(model, valid_loader, valid_samples, batch_size, loss_ftn, no_E)
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
