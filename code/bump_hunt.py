"""
Generate graphs for bump hunting on invariant mass.
"""
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import os.path as osp
import models
from graph_data import GraphDataset
from torch_geometric.data import Data, DataListLoader, Batch
from torch.utils.data import random_split
from torch.nn import MSELoss
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
import sys

def invariant_mass(jet1_e, jet1_px, jet1_py, jet1_pz, jet2_e, jet2_px, jet2_py, jet2_pz):
    """
        Calculates the invariant mass between 2 jets. Based on the formula:
        m_12 = sqrt((E_1 + E_2)^2 - (p_x1 + p_x2)^2 - (p_y1 + p_y2)^2 - (p_z1 + p_z2)^2)

        Args:
            jet1_(e, px, py, pz) (torch.float): 4 momentum of first jet of dijet
            jet2_(e, px, py, pz) (torch.float): 4 momentum of second jet of dijet

        Returns:
            torch.float dijet invariant mass.
    """
    return torch.sqrt(torch.square(jet1_e + jet2_e) - torch.square(jet1_px + jet2_px)
                      - torch.square(jet1_py + jet2_py) - torch.square(jet1_pz + jet2_pz))

def sparseloss3d(x,y):
    """
    Sparse loss function for autoencoders, a permutation invariant euclidean distance function
    from set x -> y and y -> x.

    Args:
        x (torch.tensor): input sample
        y (torch.tensor): output sample

    Returns:
        torch.tensor of the same shape as x and y representing the loss.
    """
    num_parts = x.shape[0]
    dist = torch.pow(torch.cdist(x,y),2)
    in_dist_out = torch.min(dist,dim=0)
    out_dist_in = torch.min(dist,dim=1)
    loss = torch.sum(in_dist_out.values + out_dist_in.values) / num_parts
    return loss

def make_bump_graph(nonoutlier_mass, outlier_mass, x_lab, save_name, bins, output_dir):
    """
    Create matplotlib graphs, overlaying histograms of invariant mass for outliers and all events.

    Args:
        nonoutlier_mass (tensor): inv mass of jets from nonoutlier event
        outlier_mass (tensor): inv mass from outlier events
        x_lab (str): x axis label for graph
        save_name (str): what name to save graph pdf as
        bins (np.linspace): the bins for the histogram
    """
    # plot mjj bump histograms
    plt.figure(figsize=(6,4.4))
    weights = np.ones_like(outlier_mass) / len(outlier_mass)
    plt.hist(outlier_mass, alpha = 0.5, bins=bins, weights=weights, label='Outlier events')
    weights = np.ones_like(nonoutlier_mass) / len(nonoutlier_mass)
    plt.hist(nonoutlier_mass, alpha = 0.5, bins=bins, weights=weights, label='Nonoutlier events')
    plt.legend()
    plt.xlabel(x_lab, fontsize=16)
    plt.ylabel('Normalized events [a. u.]', fontsize=16)
    plt.tight_layout()
    plt.savefig(osp.join(output_dir,save_name+'.pdf'))
    plt.close()

def make_loss_graph(losses, save_name, output_dir):
    plt.figure(figsize=(6,4.4))
    plt.hist(losses,bins=np.linspace(0, 600, 101))
    plt.savefig(osp.join(output_dir,save_name+'.pdf'))
    plt.xlabel('Loss', fontsize=16)
    plt.ylabel('Jets', fontsize=16)
    plt.close()

def process(data_loader, num_events, model_fname, model_num, use_sparseloss, latent_dim):
    """
    Use the specified model to determine the reconstruction loss of each sample.
    Also calculate the invariant mass of the jets.

    Args:
        data_loader (torch.data.DataLoader): pytorch dataloader for loading in black boxes
        num_events (int): how many events we're processing
        model_fname (str): name of saved model
        model_num (int): 0 for EdgeConv based models, 1 for MetaLayer based models
        use_sparseloss (bool): toggle for using sparseloss instead of mse

    Returns: torch.tensor of size (num_events, 5).
             column-wise: [jet1_loss, jet2_loss, dijet_invariant_mass, jet1_mass, jet2_mass, rnd_truth_bit]
             Row-wise: dijet of event
        
    """
    # load model for loss calculation
    model = models.EdgeNet(hidden_dim=latent_dim) # default to edgeconv network
    if model_num == 1: # use metalayer gnn instead
        model = models.GNNAutoEncoder()
    modpath = osp.join('/anomalyvol/models/',model_fname+'.best.pth')
    model.load_state_dict(torch.load(modpath))
    model.eval()
    loss_ftn = MSELoss(reduction='mean')
    # use sparseloss function instead of default mse
    if use_sparseloss:
        loss_ftn = sparseloss3d

    # Store the return values
    max_feat = 4
    jets_proc_data = []
    input_fts = []
    reco_fts = []

    # for each event in the dataset calculate the loss and inv mass for the leading 2 jets
    with torch.no_grad():
        for k, data in enumerate(data_loader):
            data = data[0] # remove extra brackets
            # mask 3rd jet in 3-jet events
            events = torch.stack([d.u[0][0] for d in data]).cpu().numpy()
            mask3jet = np.insert(np.diff(events).astype(bool), 0, True)
            mask3jet[np.insert(mask3jet[:-1].astype(bool), 0, False)] = True
            data = [d for d,m in zip(data,mask3jet) if m]
            # run inference on all jets
            data_batch = Batch.from_data_list(data)
            jets_rec = model(data_batch)
            # get first and second jets
            batch = data_batch.batch
            jets_x = data_batch.x
            jets0_x = jets_x[::2]
            jets1_x = jets_x[1::2]
            jets0_rec = jets_rec[::2]
            jets1_rec = jets_rec[1::2]
            jets_u = data_batch.u
            jets0_u = jets_u[::2]
            jets1_u = jets_u[1::2]
            # calculate invariant mass (data.u format: p[event_idx, n_particles, jet.mass, jet.px, jet.py, jet.pz, jet.e]])
            dijet_mass = invariant_mass(jets0_u[:,6], jets0_u[:,3], jets0_u[:,4], jets0_u[:,5],
                                        jets1_u[:,6], jets1_u[:,3], jets1_u[:,4], jets1_u[:,5])
            njets = len(torch.unique(batch))
            losses = torch.zeros((njets), dtype=torch.float32)
            for ib in torch.unique(batch):
                losses[ib] = loss_ftn(jets_rec[batch==ib], jets_x[batch==ib])
            loss0 = losses[::2]
            loss1 = losses[1::2]
            jets_info = torch.stack([loss0,
                                     loss1,
                                     dijet_mass,              # mass of dijet
                                     jets0_u[:,2],            # mass of jet 1
                                     jets1_u[:,2],            # mass of jet 2
                                     jets1_u[:,-1]],          # if this event was an anomaly
                                    dim=1)
            jets_proc_data.append(jets_info)
            input_fts.append(jets_x[::2])
            input_fts.append(jets_x[1::2])
            reco_fts.append(jets_rec[::2])
            reco_fts.append(jets_rec[1::2])
    # return pytorch tensors
    return torch.cat(jets_proc_data), torch.cat(input_fts), torch.cat(reco_fts)


def bump_hunt(df, cuts, model_fname, model_num, use_sparseloss, bb, output_dir):
    """
    Loops and makes multiple cuts on the jet losses, and generates a graph for each cut by
    delegating to make_bump_graph().

    Args:
        df (pd.DataFrame): output of process() transformed into datafram; has loss and mass of jets per event
        cuts (list of floats): all the percentages to perform a cut on the loss
        model_fname (str): name of saved model
        model_num (int): 0 for EdgeConv based models, 1 for MetaLayer based models
        use_sparseloss (bool): toggle for using sparseloss instead of mse
        bb (str): which black box the bump hunt is being performed on (e.g. 'bb1')
    """
    losses = np.concatenate([df['loss1'], df['loss2']])
    make_loss_graph(losses,  osp.join(savedir,'loss_distribution'), output_dir)

    # generate a graph for different cuts
    for cut in cuts:
        # name for graph files when saved
        dijet_graph_name = ""
        mj1_graph_name = ""
        mj2_graph_name = ""
        if use_sparseloss == True:
            dijet_graph_name = savedir + 'dijet_bump_' + str(cut)
            mj1_graph_name = savedir + 'mj1_bump_' + str(cut)
            mj2_graph_name = savedir + 'mj2_bump_' + str(cut)
        else:
            dijet_graph_name = savedir + 'dijet_bump_' + str(cut)
            mj1_graph_name = savedir + 'mj1_bump_' + str(cut)
            mj2_graph_name = savedir + 'mj2_bump_' + str(cut)

        loss_thresh = np.quantile(losses, cut)
        # classify dijet as outlier if both jets are outliers
        df['outlier'] = (np.minimum(df['loss1'], df['loss2']) > loss_thresh)
        outliers = df[df['outlier']]
        # otherwise, classify dijet as nonoutlier
        nonoutliers = df[~df['outlier']]
        # alternative definition:
        #df['nonoutlier'] = (np.maximum(df['loss1'], df['loss2']) < loss_thresh)
        #nonoutliers = df[df['nonoutlier']]

        # make dijet bump hunt graph
        all_dijet_mass = df['dijet_mass']
        nonoutlier_dijet_mass = nonoutliers['dijet_mass']
        outlier_dijet_mass = outliers['dijet_mass'] # get the mass of only outliers

        x_lab = '$m_{jj}$ [GeV]'
        bins = np.linspace(1000, 6000, 51)
        make_bump_graph(nonoutlier_dijet_mass, outlier_dijet_mass, x_lab, dijet_graph_name, bins, output_dir)

        # make graph for mj1
        all_m1_mass = df['mass1']
        nonoutlier_m1_mass = nonoutliers['mass1']
        outlier_m1_mass = outliers['mass1']
        x_lab = '$m_{j1}$ [GeV]'
        bins = np.linspace(0, 1800, 51)
        make_bump_graph(nonoutlier_m1_mass, outlier_m1_mass, x_lab, mj1_graph_name, bins, output_dir)

        # make graph for mj2
        all_m2_mass = df['mass2']
        nonoutlier_m2_mass = nonoutliers['mass2']
        outlier_m2_mass = outliers['mass2']
        x_lab = '$m_{j2}$ [GeV]'
        bins = np.linspace(0, 1800, 51)
        make_bump_graph(nonoutlier_m2_mass, outlier_m2_mass, x_lab, mj2_graph_name, bins, output_dir)

    if bb == 'rnd':  # plot roc for rnd set
        df['loss_sum'] = df['loss1']+df['loss2']
        df['loss_min'] = np.minimum(df['loss1'],df['loss2'])
        df['loss_max'] = np.maximum(df['loss1'],df['loss2'])

        plt.figure(figsize=(6,4.4))
        lw = 2            
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

        for var in ['sum', 'min', 'max']:
            fpr, tpr, thresholds = metrics.roc_curve(df['truth_bit'],df['loss_'+var])
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, 
                     lw=lw, label='ROC curve loss %s (area = %0.2f)' % (var,auc))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.legend(loc="lower right")
        plt.savefig(osp.join(output_dir,savedir+'_roc.pdf'))
        plt.close()

def plot_reco_difference(input_fts, reco_fts, model_fname, bb, output_dir):
    """
    Plot the difference between the autoencoder's reconstruction and the original input

    Args:
        input_fts (torch.tensor): the original features of the particles
        reco_fts (torch.tensor): the reconstructed features
        model_fname (str): name of saved model
        bb (str): which black box the input came from
    """
    Path(osp.join(output_dir,model_fname,'reconstruction')).mkdir(exist_ok=True) # make a folder for these graphs
    label = ['$p_x~[GeV]$', '$p_y~[GeV]$', '$p_z~[GeV]$', '$E~[GeV]$']
    feat = ['px', 'py', 'pz' , 'E']

    # make a separate plot for each feature
    for i in range(input_fts.shape[1]):
        plt.figure(figsize=(6,4.4))
        bins = np.linspace(-20, 20, 101)
        if i == 3:  # different bin size for E momentum
            bins = np.linspace(-5, 35, 101)
        plt.hist(input_fts[:,i], bins=bins, alpha=0.5, label='input')
        plt.hist(reco_fts[:,i], bins=bins, alpha=0.5, label='output')
        plt.legend()
        plt.xlabel(label[i], fontsize=16)
        plt.ylabel('Particles', fontsize=16)
        plt.savefig(osp.join(output_dir, model_fname, 'reconstruction', feat[i] + '_' + bb + '.pdf'))
        plt.close()

    
if __name__ == "__main__":
    # process arguments
    import argparse
    parser = argparse.ArgumentParser()
    print("model_name options:")
    saved_models = [osp.basename(x)[:-9] for x in glob.glob('/anomalyvol/models/*')]
    print(saved_models)
    parser.add_argument("--model_name", type=str, help="Saved model name discluding file extension.", required=True, choices=saved_models)
    parser.add_argument("--output_dir", type=str, help="Output directory for files.", required=False, default='/anomalyvol/figures/')
    parser.add_argument("--model_num", type=int, help="0 = EdgeConv, 1 = MetaLayer", required=True)
    parser.add_argument("--use_sparseloss", action='store_true', help="Toggle use of sparseloss. Default False.", default=False, required=False)
    parser.add_argument("--overwrite", action='store_true', help="Toggle overwrite of pkl. Default False.", default=False, required=False)
    parser.add_argument("--num_events", type=int, help="How many events to process (multiple of 100). Default 1mil", default=1000000, required=False)
    parser.add_argument("--latent_dim", type=int, help="How many units for the latent space (def=2)", default=2, required=False)
    args = parser.parse_args()

    # validate arguments
    if args.num_events <= 0 or args.num_events > 1000000:
        exit("--num_events must be in range (0, 1000000]")
    if args.model_num not in [0, 1]:
        exit("--model_num can only be 0 (EdgeNet) or 1 (MetaLayer)")
    if args.latent_dim <= 0:
        exit("--latent_dim must be greater than 0")
    model_fname = args.model_name
    model_num = args.model_num
    use_sparseloss = args.use_sparseloss
    num_events = args.num_events
    latent_dim = args.latent_dim
    output_dir = args.output_dir
    overwrite = args.overwrite

    Path(output_dir).mkdir(exist_ok=True) # make a folder for the graphs of this model   
    Path(osp.join(output_dir,model_fname)).mkdir(exist_ok=True) # make a folder for the graphs of this model

    cuts = np.arange(0.2, 1.0, 0.1)

    num_files = int(10000 - (10000 * (1100000 - num_events) / 1100000)) # how many files to read
    if num_events == 1000000:    # account for fact rnd set has 100k more events then other boxes
        num_files = int(10000 - (10000 * (1100000 - (num_events + 100000)) / 1100000)) # how many files to read
    ignore_files = 10000 - num_files


    def get_df(proc_jets):
        d = {'loss1': proc_jets[:,0],
             'loss2': proc_jets[:,1],
             'dijet_mass': proc_jets[:,2],
             'mass1': proc_jets[:,3],
             'mass2': proc_jets[:,4],
             'truth_bit': proc_jets[:,5]}
        df = pd.DataFrame(d)
        return df
    
    print("Plotting RnD set")
    savedir = osp.join(model_fname, 'rnd')
    Path(osp.join(output_dir,savedir)).mkdir(exist_ok=True) # make a subfolder
    bb4 = GraphDataset('/anomalyvol/data/lead_2/rnd/', bb=4)
    torch.manual_seed(0) # consistency for random_split
    bb4, ignore, ignore2 = random_split(bb4, [num_files, ignore_files, 0])
    bb4_loader = DataListLoader(bb4)
    if not osp.isfile(osp.join(output_dir,model_fname,'rnd','df.pkl')) or overwrite:
        proc_jets, input_fts, reco_fts = process(bb4_loader, num_events, model_fname, model_num, use_sparseloss, latent_dim) # colms: loss1, loss2, dijet_m, jet1_m, jet2_m
        df = get_df(proc_jets)
        df.to_pickle(osp.join(output_dir,model_fname,'rnd','df.pkl'))
    else:
        df = pd.read_pickle(osp.join(output_dir,model_fname,'rnd','df.pkl'))
    #plot_reco_difference(input_fts, reco_fts, model_fname, 'bb4', output_dir)  # plot reconstruction difference
    bump_hunt(df, cuts, model_fname, model_num, use_sparseloss, 'rnd', output_dir)  # plot bump hunts

    num_files = int(10000 - (10000 * (1000000 - num_events) / 1000000)) # how many files to read
    ignore_files = 10000 - num_files
    for i, bb_name in enumerate(["bb0", "bb1", "bb2"]):
        print("Plotting %s"%bb_name)
        savedir = osp.join(model_fname, bb_name)
        Path(osp.join(output_dir,savedir)).mkdir(exist_ok=True) # make a subfolder
        bb = GraphDataset('/anomalyvol/data/lead_2/%s/'%bb_name, bb=i)
        bb, ignore, ignore2 = random_split(bb, [num_files, ignore_files, 0])
        bb_loader = DataListLoader(bb)
        if not osp.isfile(osp.join(output_dir,model_fname,bb_name,'df.pkl')) or overwrite:
            proc_jets, input_fts, reco_fts = process(bb_loader, num_events, model_fname, model_num, use_sparseloss, latent_dim) # colms: loss1, loss2, dijet_m, jet1_m, jet2_m
            df = get_df(proc_jets)
            df.to_pickle(osp.join(output_dir,model_fname,bb_name,'df.pkl'))
        else:
            df = pd.read_pickle(osp.join(output_dir,model_fname,bb_name,'df.pkl'))
        #plot_reco_difference(input_fts, reco_fts, model_fname, bb_name, output_dir)
        bump_hunt(df, cuts, model_fname, model_num, use_sparseloss, bb_name, output_dir)
