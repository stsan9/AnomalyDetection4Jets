"""
Generate Plots for Bump Hunting
"""
import glob
import matplotlib.pyplot as plt
import torch
import os.path as osp
import models
from graph_data import GraphDataset
from torch_geometric.data import Data, DataListLoader, Batch
from torch.utils.data import random_split
from torch.nn import MSELoss
import numpy as np
import pandas as pd

cuts = [0.97, 0.99, 0.997, 0.999]  # loss thresholds percentiles
model_fname = ""
model_num = 0
use_sparseloss = False

# m_12 = sqrt ( (E_1 + E_2)^2 - (p_x1 + p_x2)^2 - (p_y1 + p_y2)^2 - (p_z1 + p_z2)^2 )
def invariant_mass(jet1_e, jet1_px, jet1_py, jet1_pz, jet2_e, jet2_px, jet2_py, jet2_pz):
    return torch.sqrt(torch.square(jet1_e + jet2_e) - torch.square(jet1_px + jet2_px)
                      - torch.square(jet1_py + jet2_py) - torch.square(jet1_pz + jet2_pz))

# sparseloss function
def sparseloss3d(x,y):
    nparts = x.shape[0]
    dist = torch.pow(torch.cdist(x,y),2)
    in_dist_out = torch.min(dist,dim=0)
    out_dist_in = torch.min(dist,dim=1)
    loss = torch.sum(in_dist_out.values + out_dist_in.values) / nparts
    return loss

# creates matplotlib graphs
def make_graph(all_mass, outlier_mass, bb, cut):
    # plot mjj bump histograms
    plt.figure(figsize=(6,4.4))
    bins = np.linspace(1000, 6000, 51)
    weights = np.ones_like(outlier_mass) / len(outlier_mass)
    plt.hist(outlier_mass, alpha = 0.5, bins=bins, weights=weights, label='Outlier events')
    weights = np.ones_like(all_mass) / len(all_mass)
    plt.hist(all_mass, alpha = 0.5, bins=bins, weights=weights, label='All events')
    plt.legend()
    plt.xlabel('$m_{jj}$ [GeV]', fontsize=16)
    plt.ylabel('Normalized events [a. u.]', fontsize=16)
    plt.tight_layout()
    if use_sparseloss == True:
        plt.savefig('/anomalyvol/figures/' + model_fname + '_withsparseloss_bump_' + bb + '_' + str(cut) + '.pdf')
    else:
        plt.savefig('/anomalyvol/figures/' + model_fname + '_bump_' + bb + '_' + str(cut) + '.pdf')

# loop through dataset to extract useful information
def process(data_loader, num_events):
    # load model for loss calculation
    model = models.EdgeNet() # default to edgeconv network
    if model_num == 2: # use metalayer gnn instead
        model = models.GNNAutoEncoder()
    modpath = osp.join('/anomalyvol/models/',model_fname+'.best.pth')
    model.load_state_dict(torch.load(modpath))
    model.eval()
    loss_ftn = MSELoss(reduction='mean')
    if use_sparseloss == True: # use sparseloss function instead of default mse
        loss_ftn = sparseloss3d

    # colms: e_1, px_1, py_1, pz_1, e2, px_2, py_2, pz_2, loss_1, loss_2
    # indices: 0, 1,  , 2   , 3   , 4 , 5   , 6   , 7   , 8     , 9
    jet_data = torch.zeros((num_events, 5), dtype=torch.float32)
    event = -1
    with torch.no_grad():
        for k, data in enumerate(data_loader): # go through all 10k data lists
            data = data[0] # remove extra brackets
            for i in range(0,len(data) - 1):    # traverse list
                event += 1
                if (event)%1000==0: print ('processing event %i'% event)
                # check that they are from the same event
                if i<len(data)-1 and data[i].u[0][0].item() != data[i+1].u[0][0].item():
                    event -= 1
                    continue
                # and that's not a 2nd+3rd jet:
                if i>0 and data[i-1].u[0][0].item() == data[i].u[0][0].item():
                    event -= 1
                    continue                    
                # run inference on both jets at the same time
                if use_sparseloss == True: # for no padding model
                    jet_0 = data[i]
                    jet_1 = data[i + 1]
                    jet_x_0 = jet_0.x
                    jet_x_1 = jet_1.x
                    jet_rec_0 = model(jet_0)
                    jet_rec_1 = model(jet_1)
                    # calculate invariant mass (data.u format: p[event_idx, n_particles, jet.mass, jet.px, jet.py, jet.pz, jet.e]])
                    jet0_u = data[i].u[0]
                    jet1_u = data[i+1].u[0]
                    dijet_mass = invariant_mass(jet0_u[6], jet0_u[3], jet0_u[4], jet0_u[5],
                                                jet1_u[6], jet1_u[3], jet1_u[4], jet1_u[5])
                    jet_losses = torch.tensor([loss_ftn(jet_rec_0, jet_x_0), # loss of jet 1
                                               loss_ftn(jet_rec_1, jet_x_1), # loss of jet 2
                                               dijet_mass,              # mass of dijet
                                               jet0_u[2],               # mass of jet 1
                                               jet1_u[2]])              # mass of jet 2
                    jet_data[event,:] = jet_losses
                else:
                    jets = Batch.from_data_list(data[i:i+2])
                    jets_x = jets.x
                    jets_rec = model(jets)
                    jet_rec_0 = jets_rec[jets.batch==jets.batch[0]]
                    jet_rec_1 = jets_rec[jets.batch==jets.batch[-1]]
                    jet_x_0 = jets_x[jets.batch==jets.batch[0]]
                    jet_x_1 = jets_x[jets.batch==jets.batch[-1]]
                    # calculate invariant mass (data.u format: p[event_idx, n_particles, jet.mass, jet.px, jet.py, jet.pz, jet.e]])
                    jet0_u = data[i].u[0]
                    jet1_u = data[i+1].u[0]
                    dijet_mass = invariant_mass(jet0_u[6], jet0_u[3], jet0_u[4], jet0_u[5],
                                                jet1_u[6], jet1_u[3], jet1_u[4], jet1_u[5])
                    jet_losses = torch.tensor([loss_ftn(jet_rec_0, jet_x_0), # loss of jet 1
                                               loss_ftn(jet_rec_1, jet_x_1), # loss of jet 2
                                               dijet_mass,              # mass of dijet
                                               jet0_u[2],               # mass of jet 1
                                               jet1_u[2]])              # mass of jet 2
                    jet_data[event,:] = jet_losses
                
                
    return jet_data[:event] # cut off extra zeros if any

# Integrate all parts
def bump_hunt(num_events):
    num_files = int(10000 - (10000 * (1000000 - num_events) / 1000000)) # how many files to read
    ignore_files = 10000 - num_files
    torch.manual_seed(0)
    
    print("Plotting bb1")
    bb1 = GraphDataset('/anomalyvol/data/gnn_node_global_merge/bb1/', bb=1)
    bb1, ignore, ignore2 = random_split(bb1, [num_files, ignore_files, 0])
    print("done processing bb1")
    bb1_loader = DataListLoader(bb1)
    jet_losses = process(bb1_loader, num_events) # colms: [jet1_loss, jet2_loss]
    losses = jet_losses[:,:2].flatten().numpy()
    for cut in cuts:
        loss_thresh = np.quantile(losses, cut)
        d = {'loss1': jet_losses[:,0],
             'loss2': jet_losses[:,1],
             'dijet_mass': jet_losses[:,2],
             'mass1': jet_losses[:,3],
             'mass2': jet_losses[:,4]}
        df = pd.DataFrame(d)
        all_dijet_mass = df['dijet_mass']
        # id outliers
        df['outlier'] = 0
        df.loc[(df['loss1'] > loss_thresh) | (df['loss2'] > loss_thresh), 'outlier'] = 1
        outliers = df.loc[df.outlier == 1]
        # get the mass of only outliers
        outlier_dijet_mass = outliers['dijet_mass']
        # make graph
        make_graph(all_dijet_mass, outlier_dijet_mass, 'bb1', cut)
    
    print("Plotting bb2")
    bb2 = GraphDataset('/anomalyvol/data/gnn_node_global_merge/bb2/', bb=2)
    bb2, ignore, ignore2 = random_split(bb2, [num_files, ignore_files, 0])
    bb2_loader = DataListLoader(bb2)
    jet_losses = process(bb2_loader, num_events) # colms: [jet1_loss, jet2_loss]
    losses = jet_losses[:,:2].flatten().numpy()
    for cut in cuts:
        loss_thresh = np.quantile(losses, cut)
        d = {'loss1': jet_losses[:,0],
             'loss2': jet_losses[:,1],
             'dijet_mass': jet_losses[:,2],
             'mass1': jet_losses[:,3],
             'mass2': jet_losses[:,4]}
        df = pd.DataFrame(d)
        all_dijet_mass = df['dijet_mass']
        # id outliers
        df['outlier'] = 0
        df.loc[(df['loss1'] > loss_thresh) | (df['loss2'] > loss_thresh), 'outlier'] = 1
        outliers = df.loc[df.outlier == 1]
        # get the mass of only outliers
        outlier_dijet_mass = outliers['dijet_mass']
        # make graph
        make_graph(all_dijet_mass, outlier_dijet_mass, 'bb2', cut)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    print("Model name options:")
    print([osp.basename(x)[:-9] for x in glob.glob('/anomalyvol/models/*')])
    parser.add_argument("--model_name", type=str, help="saved model name discluding file extension", required=True)
    parser.add_argument("--model_num", type=int, help="1 = EdgeConv, 2 = MetaLayer", required=True)
    parser.add_argument("--use_sparseloss", type=bool, help="Boolean toggle use sparseloss (default False)", required=False)
    parser.add_argument("--num_events", type=int, help="how many events to process (multiple of 100)", required=True)
    args = parser.parse_args()
    
    use_sparseloss = args.use_sparseloss
    model_num = args.model_num
    model_fname = args.model_name
    if model_num > 0 and model_num <= 3:
        bump_hunt(args.num_events)
    else:
        print("Invalid model_num. Can only be 1 (EdgeNet) or 2 (MetaLayer)")
