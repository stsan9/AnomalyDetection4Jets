import torch
import tqdm
import math
import torch.nn as nn
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from torch.utils.data import random_split
import os.path as osp
from graph_data import GraphDataset
import models

# darkflow loss function
def sparseloss3d(x,y):
    nparts = x.shape[0]
    dist = torch.pow(torch.cdist(x,y),2)
    in_dist_out = torch.min(dist,dim=0)
    out_dist_in = torch.min(dist,dim=1)
    loss = torch.sum(in_dist_out.values + out_dist_in.values) / nparts
    return loss

# Reconstruction + KL divergence losses
def vae_loss(x, y, mu, logvar):
    BCE = sparseloss3d(x,y)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# train and test helper functions
@torch.no_grad()
def test(model, loader, total, batch_size, no_E = False, use_sparseloss = False):
    model.eval()
    
    loss_ftn = nn.MSELoss(reduction='mean')
    if (use_sparseloss == True):
        loss_ftn = sparseloss3d

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        if (no_E == True):
            data.x = data.x[:,:-1]
        y = data.x # the model will overwrite data.x, so save a copy
        y = y.contiguous()
        batch_output = model(data)
        batch_loss_item = loss_ftn(batch_output, y).item()
        sum_loss += batch_loss_item
        t.set_description("loss = %.5f" % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def train(model, optimizer, loader, total, batch_size, no_E = False, use_sparseloss = False):
    model.train()
    
    loss_ftn = nn.MSELoss(reduction='mean')
    if (use_sparseloss == True):
        loss_ftn = sparseloss3d

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        if (no_E == True):
            data.x = data.x[:,:-1]
        y = data.x # the model will overwrite data.x, so save a copy
        y = y.contiguous()
        optimizer.zero_grad()
        batch_output = model(data)
        batch_loss = loss_ftn(batch_output, y)
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description("loss = %.5f" % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()
    
    return sum_loss/(i+1)

# variations of test() and train() for sparseloss (no padding; stochastic gradient descent)
@torch.no_grad()
def single_steps_test(model, loader, total, batch_size, no_E = False, use_sparseloss = True, use_vae = False):
    model.eval()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data_list in t:
        for data in data_list:
            if data.x.shape[0] <= 1:
                continue
            data = data.to(device)
            if (no_E == True):
                data.x = data.x[:,:-1]
            y = data.x
            y = y.contiguous()
            if use_vae == True:
                batch_output, mu, log_var = model(data)
                batch_loss_item = vae_loss(batch_output, y, mu, log_var).item()
            elif use_sparseloss == True:
                batch_output = model(data)
                batch_loss_item = sparseloss3d(batch_output, y).item()
            sum_loss += batch_loss_item
            t.set_description("loss = %.5f" % (batch_loss_item))
            t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def sgd_train(model, optimizer, loader, total, batch_size, no_E = False, use_sparseloss = True, use_vae = False):
    model.train()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data_list in t:
        for data in data_list:
            if data.x.shape[0] <= 1:
                continue
            data = data.to(device)
            if (no_E == True):
                data.x = data.x[:,:-1]
            y = data.x
            y = y.contiguous()
            optimizer.zero_grad()
            if use_vae == True:
                batch_output, mu, log_var = model(data)
                batch_loss = vae_loss(batch_output, y, mu, log_var)
            elif use_sparseloss == True:
                batch_output = model(data)
                batch_loss = sparseloss3d(batch_output, y)
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
    parser.add_argument("--lat_dim", type=int, help="latent space size", default=2, required=False)
    parser.add_argument("--no_E", action='store_true', help="Toggle to remove energy from training and testing. Default False.", default=False, required=False)
    parser.add_argument("--mod_name", type=str, help="model name for saving and loading", required=True)
    parser.add_argument("--sparseloss", action='store_true', help="Toggle use of sparseloss. Default False.", default=False, required=False)
    parser.add_argument("--metalayer", action='store_true', help="Toggle to use metalayer model. Defaulted to edgeconv.", default=False, required=False)
    parser.add_argument("--vae", action='store_true', help="Toggle to use vae edgeconv model. Defaulted to edgeconv.", default=False, required=False)
    parser.add_argument("--box_num", type=int, help="0=QCD-background; 1=bb1; 2=bb2; 4=rnd", default=0, required=False)
    args = parser.parse_args()
    # data and specifications
    if args.box_num == 0:
        gdata = GraphDataset(root='/anomalyvol/data/gnn_node_global_merge', bb=0) 
    elif args.box_num == 1:
        gdata = GraphDataset(root='/anomalyvol/data/bb_train_sets/bb1', bb=1) 
    elif args.box_num == 2:
        gdata = GraphDataset(root='/anomalyvol/data/bb_train_sets/bb2', bb=2) 
    elif args.box_num == 4:
        gdata = GraphDataset(root='/anomalyvol/data/rnd_set', bb=4)
    use_sparseloss = args.sparseloss
    use_vae = args.vae
    no_E = args.no_E
    input_dim = 3 if args.no_E else 4
    big_dim = 32
    hidden_dim = args.lat_dim
    fulllen = len(gdata)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    batch_size = 2
    n_epochs = 200
    lr = 0.001
    patience = 10
    device = 'cuda:0'
    model_fname = args.mod_name

    if args.metalayer:
        print("Using metalayer model")
        model = models.GNNAutoEncoder().to(device)
    elif args.vae:
        print("Using VAE")
        model = models.EdgeNetVAE(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim).to(device)
    else:
        print("Using default EdgeConv")
        model = models.EdgeNet(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    def collate(items): # collate function for data loaders (transforms list of lists to list)
        l = sum(items, [])
        return Batch.from_data_list(l)

    # train, valid, test split
    torch.manual_seed(0) # lock seed for random_split
    train_dataset, valid_dataset, test_dataset = random_split(gdata, [fulllen-2*tv_num,tv_num,tv_num])
    train_loader = -1
    valid_loader = -1
    test_loader = -1

    if use_sparseloss == False and use_vae == False:
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

    # load in model
    modpath = osp.join('/anomalyvol/models/',model_fname+'.best.pth')
    try:
        if torch.cuda.is_available():
            print("Using GPU")
            model.load_state_dict(torch.load(modpath, map_location=torch.device('cuda')))
        else:
            model.load_state_dict(torch.load(modpath, map_location=torch.device('cpu')))
    except:
        pass # new model

    # Training loop
    stale_epochs = 0
    best_valid_loss = 9999999
    loss = best_valid_loss
    if use_sparseloss == False and use_vae == False:
        best_valid_loss = test(model, valid_loader, valid_samples, batch_size, no_E, use_sparseloss, use_vae)
    else:
        best_valid_loss = single_steps_test(model, valid_loader, valid_samples, batch_size, no_E, use_sparseloss, use_vae)

    for epoch in range(0, n_epochs):
        if use_sparseloss == False and use_vae == False:
            loss = train(model, optimizer, train_loader, train_samples, batch_size, no_E, use_sparseloss)
            valid_loss = test(model, valid_loader, valid_samples, batch_size, no_E, use_sparseloss)
        else:
            loss = sgd_train(model, optimizer, train_loader, train_samples, batch_size, no_E, use_sparseloss, use_vae)
            valid_loss = single_steps_test(model, valid_loader, valid_samples, batch_size, no_E, use_sparseloss, use_vae)
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
