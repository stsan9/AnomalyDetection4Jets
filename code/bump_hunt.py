"""
Generate Plots for Bump Hunting
"""
import matplotlib as plt
import torch
import os.path as osp
from models import EdgeNet
from graph_data import GraphDataset
from torch_geometric.data import Data, DataListLoader, Batch
from torch.nn import MSELoss
import numpy as np

cuts = 0.97  # loss thresholds percentiles
model_fname = "GNN_AE_EdgeConv" # default
#batch_size = 4


# m_12 = sqrt ( (E_1 + E_2)^2 - (p_x1 + p_x2)^2 - (p_y1 + p_y2)^2 - (p_z1 + p_z2)^2 )
def invariant_mass(jet1_e, jet1_px, jet1_py, jet1_pz, jet2_e, jet2_px, jet2_py, jet2_pz):
    return torch.sqrt((jet1_e + jet2_e)**2 - (jet1_px + jet2_px)**2 - (jet1_py + jet2_py)**2 - (jet1_pz + jet2_pz)**2)

def collate(items):
    l = sum(items, [])
    return Batch.from_data_list(l)

def classify_outlier(loss1, loss2, thresh):
    # classify as outlier if either jet in dijet pairing is greater than threshold
    if loss1 >= thresh or loss2 >= thresh:
        return 1.
    return 0.

# return list of outlier masses
def outlier_mass_distinction(data):
    outlier_mass = torch.empty(0)
    for i, dijet in enumerate(data):
        mass = dijet[0]
        outlier_class = dijet[1]
        if outlier_class == 1:
            outlier_mass = torch.cat((outlier_mass, mass))
    return outlier_mass

# heavy lifting: find the invariant masses with outlier distinction
def process(data_loader, data_len):
    # load model for loss calculation
    model = EdgeNet()
    modpath = osp.join('/anomalyvol/models/',model_fname+'.best.pth')
    model.load_state_dict(torch.load(modpath))
    model.eval()
    mse = MSELoss(reduction='mean')

    # colms: e_1, px_1, py_1, pz_1, e2, px_2, py_2, pz_2, loss_1, loss_2
    # indices: 0, 1,  , 2   , 3   , 4 , 5   , 6   , 7   , 8     , 9
    jet_data = torch.empty(0, 10, dtype=torch.float32)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if(len(data) >= 2):
                jet1 = data[0]
                jet2 = data[1]
                # calculate loss
                jet1_rec = model(jet1)
                jet1_y = jet1.x
                loss1 = mse(jet1_rec, jet1_y)
                jet2_rec = model(jet2)
                jet2_y = jet2.x
                loss2 = mse(jet2_rec, jet2_y)
                # data.u = ([n_particles, jet.mass, jet.px, jet.py, jet.pz, jet.e], dtype=torch.float)
                dijet = torch.tensor([[jet1.u[5], jet1.u[2], jet1.u[3], jet1.u[4],
                                       jet2.u[5], jet2.u[2], jet2.u[3], jet2.u[4],
                                       loss1, loss2]])
                jet_data = torch.cat((jet_data, dijet))
    
    # calc invariant mass for all jets
    all_mass = invariant_mass(jet_data[:,0], jet_data[:,1], jet_data[:,2], jet_data[:,3], # jet1 : e, px, py, pz
                              jet_data[:,4], jet_data[:,5], jet_data[:,6], jet_data[:,7]) # jet2 : e, px, py, pz
    all_mass = all_mass.view(-1, 1) # change to 2d column
    jet_data = torch.cat((jet_data, all_mass), 1) # columnwise concatenation
    
    # identify outliers (get threshold and then filter)
    jet_data = jet_data[:,8:] # cut off four-momentum info-> [loss1, loss2, mass]
    losses = jet_data[:,:-1].flatten().numpy()
    mse_thresh = np.quantile(losses, cut)
    outliers = classify_outlier(jet_data[:,0], jet_data[:,1], mse_thresh) # jet1_loss, jet2_loss, thresh
    jet_data = torch.cat((jet_data[:,2], outliers), 1) # colms: [mass, outlier_class]
    outlier_mass = outlier_mass_distinction(jet_data)
    return all_mass, outlier_mass

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

# Integrate all parts
def bump_hunt():
    print("Plotting bb1")
    bb1 = GraphDataset('/anomalyvol/data/gnn_node_global_merge/', bb=1)
    bb1_loader = DataListLoader(bb1)
    bb1.collate_fn = collate
    bb1_size = len(bb1)
    all_mass, outlier_mass = process(bb1_loader, bb1_size)
    make_graph(all_mass, outlier_mass, 'bb1')
    
    print("Plotting bb2")
    bb2 = GraphDataset('/anomalyvol/data/gnn_geom/bb2/', bb=2)
    bb2_loader = DataListLoader(bb2)
    bb2.collate_fn = collate
    bb2_size = len(bb1)
    all_mass, outlier_mass = process(bb2_loader, bb2_size)
    make_graph(all_mass, outlier_mass, 'bb2')

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, help="saved modelname discluding file extension", required=False)
    args = parser.parse_args()
    
    # model_fname = args.modelname
    bump_hunt()