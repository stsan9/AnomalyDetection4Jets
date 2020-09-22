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

# train and test helper functions
@torch.no_grad()
def test(model,loader,total,batch_size, no_E = False, use_sparseloss = False):
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat_dim", type=int, help="latent space size", default=2, required=False)
    parser.add_argument("--no_E", type=int, help="Toggle to remove energy from training and testing (0: False, 1: True)", default=0, required=False)
    parser.add_argument("--mod_name", type=str, help="model name for saving and loading", required=True)
    parser.add_argument("--use_sparseloss", type=int, help="Toggle to use sparseloss (0: False, 1: True)", default=0, required=False)
    parser.add_argument("--use_metalayer", type=int, help="Toggle to use metalayer model; defaulted to edgenet (0: edgenet, 1: metalayer)", default=0, required=False)
    args = parser.parse_args()
    # data and specifications
    gdata = GraphDataset(root='/anomalyvol/data/gnn_node_global_merge', bb=0)
    use_sparseloss = [False, True][args.use_sparseloss]
    no_E = [False, True][args.no_E]
    input_dim = 3 if args.no_E else 4
    big_dim = 32
    hidden_dim = args.lat_dim
    fulllen = len(gdata)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    batch_size = 4
    n_epochs = 100
    lr = 0.001
    patience = 10
    device = 'cuda:0'
    model_fname = args.mod_name

    model = models.EdgeNet(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim).to(device)
    if args.use_metalayer:
        model = models.GNNAutoEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    def collate(items): # collate function for data loaders (transforms list of lists to list)
        l = sum(items, [])
        return Batch.from_data_list(l)

    # train, valid, test split
    torch.manual_seed(0) # lock seed for random_split
    train_dataset, valid_dataset, test_dataset = random_split(gdata, [fulllen-2*tv_num,tv_num,tv_num])
    train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    train_loader.collate_fn = collate
    valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    valid_loader.collate_fn = collate
    test_loader = DataListLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader.collate_fn = collate

    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)
    test_samples = len(test_dataset)

    # load in model
    modpath = osp.join('/anomalyvol/models/',model_fname+'.best.pth')
    try:
        model.load_state_dict(torch.load(modpath))
    except:
        pass

    # Training loop
    stale_epochs = 0
    best_valid_loss = test(model, valid_loader, valid_samples, batch_size, no_E, use_sparseloss)
    for epoch in range(0, n_epochs):
        loss = train(model, optimizer, train_loader, train_samples, batch_size, no_E, use_sparseloss)
        valid_loss = test(model, valid_loader, valid_samples, batch_size, no_E, use_sparseloss)
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
