"""
Generate Plots for Bump Hunting
"""
import matplotlib.pyplot as plt
import torch
import os.path as osp
from models import EdgeNet
from graph_data import GraphDataset
from torch_geometric.data import Data, DataListLoader, Batch
from torch.nn import MSELoss
import numpy as np
import pandas as pd

cut = 0.97  # loss thresholds percentiles
model_fname = "GNN_AE_EdgeConv" # default
#batch_size = 4

def make_graph(all_mass, outlier_mass, bb):
    # plot mjj bump histograms
    plt.figure(figsize=(6,4.4))
    bins = np.linspace(1000, 6000, 51)
    weights = np.ones_like(outlier_mass) / len(outlier_mass)
    plt.hist(np.array(outlier_mass), alpha = 0.5, bins=bins, weights=weights, label='Outlier events')
    weights = np.ones_like(all_mass) / len(all_mass)
    plt.hist(np.array(all_mass), alpha = 0.5, bins=bins, weights=weights, label='All events')
    plt.legend()
    plt.xlabel('$m_{jj}$ [GeV]', fontsize=16)
    plt.ylabel('Normalized events [a. u.]', fontsize=16)
    plt.tight_layout()
    plt.savefig('/anomalyvol/figures/bump_' + bb + '.pdf')

# loop through dataset to extract useful information
def process(data_loader, data_len):
    # load model for loss calculation
    model = EdgeNet()
    modpath = osp.join('/anomalyvol/models/',model_fname+'.best.pth')
    model.load_state_dict(torch.load(modpath))
    model.eval()
    mse = MSELoss(reduction='mean')

    # colms: e_1, px_1, py_1, pz_1, e2, px_2, py_2, pz_2, loss_1, loss_2
    # indices: 0, 1,  , 2   , 3   , 4 , 5   , 6   , 7   , 8     , 9
    jet_data = torch.zeros((data_len, 2), dtype=torch.float32)
    event = -1
    with torch.no_grad():
        for k, data in enumerate(data_loader): # go through all 10k data lists
            data = data[0] # remove extra brackets
            for i in range(0,len(data)):    # traverse list
                event += 1
                if (event)%1000==0: print ('processing event %i'% event)
                # check that they are from the same event
                if i<len(data)-1 and data[i].u[0][0].item() != data[i+1].u[0][0].item():
                    continue
                # and that's not a 2nd+3rd jet:
                if i>0 and data[i-1].u[0][0].item() == data[i].u[0][0].item():
                    continue                    
                # run inference on both jets at the same time
                jets = Batch.from_data_list(data[i:i+2])
                jets_x = jets.x
                jets_rec = model(jets)
                jet_rec_0 = jets_rec[jets.batch==jets.batch[0]]
                jet_rec_1 = jets_rec[jets.batch==jets.batch[-1]]
                jet_x_0 = jets_x[jets.batch==jets.batch[0]]
                jet_x_1 = jets_x[jets.batch==jets.batch[-1]]
                jet_losses = torch.tensor([mse(jet_rec_0, jet_x_0),
                                           mse(jet_rec_1, jet_x_1)])
                jet_data[i:i+2,:] = jet_losses
    return jet_data

# Integrate all parts
def bump_hunt():
    print("Plotting bb1")
    bb1 = GraphDataset('/anomalyvol/data/gnn_node_global_merge/bb1/', bb=1)
    print("done processing bb1")
    bb1_loader = DataListLoader(bb1)
    bb1_size = len(bb1)
    jet_losses = process(bb1_loader, bb1_size) # colms: [jet1_loss, jet2_loss]
    losses = jet_losses.flatten().numpy()
    mse_thresh = np.quantile(losses, cut)
    # make dataframe of mass, loss, loss, outlier_class
    df = pd.read_hdf('/anomalyvol/data/dijet_mass/bb1_jet_mass.h5')
    all_mass = df['mass']
    df['loss1'] = jet_losses[:,0]
    df['loss2'] = jet_losses[:,1]
    # id outliers
    df['outlier'] = 0
    df.loc[(df['loss1'] > mse_thresh) | (df['loss2'] > mse_thresh), 'outlier'] = 1
    outliers = df.loc[df.outlier == 1]
    # get the mass of only outliers
    outlier_mass = outliers['mass']
    # make graph
    make_graph(all_mass, outlier_mass, 'bb1')
    
    print("Plotting bb2")
    bb2 = GraphDataset('/anomalyvol/data/gnn_node_global_merge/bb2/', bb=2)
    bb2_loader = DataListLoader(bb2)
    bb2_size = len(bb2)
    jet_losses = process(bb2_loader, bb1_size) # colms: [jet1_loss, jet2_loss]
    losses = jet_losses.flatten().numpy()
    mse_thresh = np.quantile(losses, cut)
    # get mass
    df = pd.read_hdf('/anomalyvol/data/dijet_mass/bb2_jet_mass.h5')
    all_mass = df['mass']
    df['loss1'] = jet_losses[:,0]
    df['loss2'] = jet_losses[:,1]
    df['outlier'] = 0
    df.loc[(df['loss1'] > mse_thresh) | (df['loss2'] > mse_thresh), 'outlier'] = 1
    outliers = df.loc[df.outlier == 1]
    outlier_mass = outliers['mass']
    make_graph(all_mass, outlier_mass, 'bb2')

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, help="saved modelname discluding file extension", required=False)
    args = parser.parse_args()
    
    #model_fname = args.modelname
    bump_hunt()
