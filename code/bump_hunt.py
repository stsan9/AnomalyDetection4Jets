"""
Generate graphs for bump hunting on invariant mass.
"""
1;95;0cimport glob
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

def make_bump_graph(all_mass, outlier_mass, x_lab, save_name, bins):
    """
    Create matplotlib graphs, overlaying histograms of invariant mass for outliers and all events.

    Args:
        all_mass (tensor): inv mass of jets from all event
        outlier_mass (tensor): inv mass from outlier events
        x_lab (str): x axis label for graph
        save_name (str): what name to save graph pdf as
        bins (np.linspace): the bins for the histogram
    """
    # plot mjj bump histograms
    plt.figure(figsize=(6,4.4))
    weights = np.ones_like(outlier_mass) / len(outlier_mass)
    plt.hist(outlier_mass, alpha = 0.5, bins=bins, weights=weights, label='Outlier events')
    weights = np.ones_like(all_mass) / len(all_mass)
    plt.hist(all_mass, alpha = 0.5, bins=bins, weights=weights, label='All events')
    plt.legend()
    plt.xlabel(x_lab, fontsize=16)
    plt.ylabel('Normalized events [a. u.]', fontsize=16)
    plt.tight_layout()
    plt.savefig('/anomalyvol/figures2/' + save_name + '.pdf')
    plt.close()

def make_loss_graph(losses, save_name):
    plt.figure(figsize=(6,4.4))
    plt.hist(losses,bins=np.linspace(0, 600, 101))
    plt.savefig('/anomalyvol/figures2/' + save_name + '.pdf')
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
    if use_sparseloss == True:
        loss_ftn = sparseloss3d

    # Store the return values
    max_feat = 4
    jets_proc_data = torch.zeros((num_events, 6), dtype=torch.float32)
    input_fts = []
    reco_fts = []
    event = -1 # event counter

    # for each event in the dataset calculate the loss and inv mass for the leading 2 jets
    with torch.no_grad():
        for k, data in enumerate(data_loader):
            data = data[0] # remove extra brackets
            for i in range(0,len(data) - 1): # traverse list of data objects
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
                if use_sparseloss == True:
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
                    jets_info = torch.tensor([loss_ftn(jet_rec_0, jet_x_0), # loss of jet 1
                                               loss_ftn(jet_rec_1, jet_x_1), # loss of jet 2
                                               dijet_mass,              # mass of dijet
                                               jet0_u[2],               # mass of jet 1
                                               jet1_u[2],               # mass of jet 2
                                               jet1_u[-1]])             # if this event was an anomaly
                    jets_proc_data[event,:] = jets_info
                    input_fts.append(jet_x_0)
                    input_fts.append(jet_x_1)
                    reco_fts.append(jet_rec_0)
                    reco_fts.append(jet_rec_1)
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
                    jets_info = torch.tensor([loss_ftn(jet_rec_0, jet_x_0), # loss of jet 1
                                               loss_ftn(jet_rec_1, jet_x_1), # loss of jet 2
                                               dijet_mass,              # mass of dijet
                                               jet0_u[2],               # mass of jet 1
                                               jet1_u[2],               # mass of jet 2
                                               jet1_u[-1]])             # if this event was an anomaly
                    jets_proc_data[event,:] = jets_info
                    input_fts.append(jet_x_0)
                    input_fts.append(jet_x_1)
                    reco_fts.append(jet_rec_0)
                    reco_fts.append(jet_rec_1)
                
    # return pytorch tensors
    return jets_proc_data[:event], torch.cat(input_fts), torch.cat(reco_fts)


def bump_hunt(proc_jets, cuts, model_fname, model_num, use_sparseloss, bb):
    """
    Loops and makes multiple cuts on the jet losses, and generates a graph for each cut by
    delegating to make_bump_graph().

    Args:
        proc_jets (torch.tensor): output of process(); has loss and mass of jets per event
        cuts (list of floats): all the percentages to perform a cut on the loss
        model_fname (str): name of saved model
        model_num (int): 0 for EdgeConv based models, 1 for MetaLayer based models
        use_sparseloss (bool): toggle for using sparseloss instead of mse
        bb (str): which black box the bump hunt is being performed on (e.g. 'bb1')
    """
    savedir = model_fname + '/' + bb + '/'
    Path('/anomalyvol/figures2/' + savedir).mkdir(exist_ok=True) # make a subfolder
    losses = proc_jets[:,:2].flatten().numpy()
    make_loss_graph(losses,  savedir + 'loss_distribution')
    cut_props = {'cut': [], 'tpr': [], 'fpr': []} # use to generate table if anoms known
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
        d = {'loss1': proc_jets[:,0],
             'loss2': proc_jets[:,1],
             'dijet_mass': proc_jets[:,2],
             'mass1': proc_jets[:,3],
             'mass2': proc_jets[:,4],
             'truth_bit': proc_jets[:,5]}
        df = pd.DataFrame(d)
        df['outlier'] = 0
        # classify dijet as outlier if either jet is an outlier
        df.loc[(df['loss1'] > loss_thresh) | (df['loss2'] > loss_thresh), 'outlier'] = 1
        outliers = df.loc[df.outlier == 1]
        positives = len(df.loc[df.truth_bit == 1])
        negatives = len(df.loc[df.truth_bit == 0])

        # make dijet bump hunt graph
        all_dijet_mass = df['dijet_mass']
        outlier_dijet_mass = outliers['dijet_mass'] # get the mass of only outliers
        if bb == 'rnd': # gather info for roc curve for rnd dataset
            num_caught_anoms = len(outliers.loc[outliers.truth_bit == 1])
            tpr = num_caught_anoms / positives
            fpr = (len(outliers) - num_caught_anoms) / negatives
            cut_props['cut'].append(cut)
            cut_props['tpr'].append(tpr)
            cut_props['fpr'].append(fpr)

        x_lab = '$m_{jj}$ [GeV]'
        bins = np.linspace(1000, 6000, 51)
        make_bump_graph(all_dijet_mass, outlier_dijet_mass, x_lab, dijet_graph_name, bins)

        # make graph for mj1
        all_m1_mass = df['mass1']
        outlier_m1_mass = outliers['mass1']
        x_lab = '$m_{j1}$ [GeV]'
        bins = np.linspace(0, 1800, 51)
        make_bump_graph(all_m1_mass, outlier_m1_mass, x_lab, mj1_graph_name, bins)

        # make graph for mj2
        all_m2_mass = df['mass2']
        outlier_m2_mass = outliers['mass2']
        x_lab = '$m_{j2}$ [GeV]'
        bins = np.linspace(0, 1800, 51)
        make_bump_graph(all_m2_mass, outlier_m2_mass, x_lab, mj2_graph_name, bins)

    if bb == 'rnd':  # plot roc for rnd set
        tpr = cut_props['tpr']
        fpr = cut_props['fpr']
        auc = metrics.auc(tpr, fpr)
        plt.figure(figsize=(6,4.4))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig('/anomalyvol/figures2/' + savedir + '_roc.pdf')
        plt.close()

def plot_reco_difference(input_fts, reco_fts, model_fname, bb):
    """
    Plot the difference between the autoencoder's reconstruction and the original input

    Args:
        input_fts (torch.tensor): the original features of the particles
        reco_fts (torch.tensor): the reconstructed features
        model_fname (str): name of saved model
        bb (str): which black box the input came from
    """
    Path('/anomalyvol/figures2/' + model_fname + '/reconstruction').mkdir(exist_ok=True) # make a folder for these graphs
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
        plt.savefig('/anomalyvol/figures2/' + model_fname  + '/reconstruction/' + feat[i] + '_' + bb + '.pdf')
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
    parser.add_argument("--use_sparseloss", action='store_true', help="Toggle use of sparseloss (0: False, 1: True). Default False.", default=False, required=False)
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

    Path(output_dir).mkdir(exist_ok=True) # make a folder for the graphs of this model   
    Path(osp.join(output_dir,model_fname)).mkdir(exist_ok=True) # make a folder for the graphs of this model
    
    print("Plotting RnD set")
    cuts = np.arange(0.90, 1.0, 0.01)
    bb4 = GraphDataset('/anomalyvol/data/lead_2/rnd/', bb=4)
    torch.manual_seed(0) # consistency for random_split
    num_files = int(10000 - (10000 * (1100000 - num_events) / 1100000)) # how many files to read
    if num_events == 1000000:    # account for fact rnd set has 100k more events then other boxes
        num_files = int(10000 - (10000 * (1100000 - (num_events + 100000)) / 1100000)) # how many files to read
    ignore_files = 10000 - num_files
    bb4, ignore, ignore2 = random_split(bb4, [num_files, ignore_files, 0])
    bb4_loader = DataListLoader(bb4)
    proc_jets, input_fts, reco_fts = process(bb4_loader, num_events, model_fname, model_num, use_sparseloss, latent_dim) # colms: loss1, loss2, dijet_m, jet1_m, jet2_m
    # plot_reco_difference(input_fts, reco_fts, model_fname, 'bb4')  # plot reconstruction difference
    bump_hunt(proc_jets, cuts, model_fname, model_num, use_sparseloss, 'rnd')  # plot bump hunts

    sys.exit()
    
    print("Plotting bb0")
    cuts = np.arange(0.99, 0.999, 0.001)
    bb0 = GraphDataset('/anomalyvol/data/lead_2/bb0/', bb=0)
    num_files = int(10000 - (10000 * (1000000 - num_events) / 1000000)) # how many files to read
    ignore_files = 10000 - num_files
    bb0, ignore, ignore2 = random_split(bb0, [num_files, ignore_files, 0])
    bb0_loader = DataListLoader(bb0)
    proc_jets, input_fts, reco_fts = process(bb0_loader, num_events, model_fname, model_num, use_sparseloss, latent_dim) # colms: loss1, loss2, dijet_m, jet1_m, jet2_m
    # plot_reco_difference(input_fts, reco_fts, model_fname, 'bb0')
    bump_hunt(proc_jets, cuts, model_fname, model_num, use_sparseloss, 'bb0')

    print("Plotting bb1")
    bb1 = GraphDataset('/anomalyvol/data/lead_2/bb1/', bb=1)
    bb1, ignore, ignore2 = random_split(bb1, [num_files, ignore_files, 0])
    bb1_loader = DataListLoader(bb1)
    proc_jets, input_fts, reco_fts = process(bb1_loader, num_events, model_fname, model_num, use_sparseloss, latent_dim)
    # plot_reco_difference(input_fts, reco_fts, model_fname, 'bb1')
    bump_hunt(proc_jets, cuts, model_fname, model_num, use_sparseloss, 'bb1')

    print("Plotting bb2")
    bb2 = GraphDataset('/anomalyvol/data/lead_2/bb2/', bb=2)
    bb2, ignore, ignore2 = random_split(bb2, [num_files, ignore_files, 0])
    bb2_loader = DataListLoader(bb2)
    proc_jets, input_fts, reco_fts = process(bb2_loader, num_events, model_fname, model_num, use_sparseloss, latent_dim)
    # plot_reco_difference(input_fts, reco_fts, model_fname, 'bb2')
    bump_hunt(proc_jets, cuts, model_fname, model_num, use_sparseloss, 'bb2')
