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
def process_loop(data_loader, data_len):
    # load model for loss calculation
    model = EdgeNet()
    modpath = osp.join('/anomalyvol/models/',model_fname+'.best.pth')
    model.load_state_dict(torch.load(modpath))
    model.eval()
    mse = MSELoss(reduction='mean')

    # colms: e_1, px_1, py_1, pz_1, e2, px_2, py_2, pz_2, loss_1, loss_2
    # indices: 0, 1,  , 2   , 3   , 4 , 5   , 6   , 7   , 8     , 9
    jet_data = torch.empty(0, 2, dtype=torch.float32)

    with torch.no_grad():
        for k, data in enumerate(data_loader): # go through all 10k data lists
            data = data[0] # remove extra brackets
            for i in range(len(data)):    # traverse horizontally
                if i % 2 == 1: # skip odd indices; data formatted s.t. every 2 sequential jets is a dijet
                    continue
                if(i + 1 < len(data)):
                    jet1 = data[i]
                    jet2 = data[i + 1]
                    # calculate loss
                    jet1_rec = model(jet1)
                    jet1_y = jet1.x
                    loss1 = mse(jet1_rec, jet1_y)
                    jet2_rec = model(jet2)
                    jet2_y = jet2.x
                    loss2 = mse(jet2_rec, jet2_y)
                    dijet_losses = torch.tensor([[loss1, loss2]])
                    jet_data = torch.cat((jet_data, dijet_losses))
    
    return jet_data

# Integrate all parts
def bump_hunt():
    print("Plotting bb1")
    bb1 = GraphDataset('/anomalyvol/data/gnn_node_global_merge/bb1/', bb=1)
    bb1_loader = DataListLoader(bb1)
    bb1_size = len(bb1)
    jet_losses = process(bb1_loader, bb1_size) # colms: [jet1_loss, jet2_loss]
    losses = jet_losses.flatten().numpy()
    mse_thresh = np.quantile(losses, cut)
    # get mass
    df = pd.read_hdf('/anomalyvol/data/dijet_mass/bb1_jet_mass.h5')
    all_mass = df['mass']
    df['loss1'] = jet_losses[:,0]
    df['loss2'] = jet_losses[:,1]
    df['outlier'] = 0
    data_df.loc[data_df['loss1'] > mse_thresh or data_df['loss2'] > mse_thresh, 'outlier'] = 1
    outliers = data_df.loc[data_df.outlier == 1]
    outlier_mass = outliers['mass']
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
    data_df.loc[data_df['loss1'] > mse_thresh or data_df['loss2'] > mse_thresh, 'outlier'] = 1
    outliers = data_df.loc[data_df.outlier == 1]
    outlier_mass = outliers['mass']
    make_graph(all_mass, outlier_mass, 'bb2')

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, help="saved modelname discluding file extension", required=False)
    args = parser.parse_args()

    #model_fname = args.modelname
    #bump_hunt()