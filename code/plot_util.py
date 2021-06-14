import numpy as np
import mplhep as hep
import os.path as osp
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use(hep.style.CMS)

def loss_distr(losses, save_name):
    """
        Plot distribution of losses
    """
    plt.figure(figsize=(6,4.4))
    plt.hist(losses,bins=np.linspace(0, 600, 101))
    plt.xlabel('Loss', fontsize=16)
    plt.ylabel('Jets', fontsize=16)
    plt.savefig(osp.join(save_name+'.pdf'))
    plt.close()

def plot_reco_difference(input_fts, reco_fts, model_fname, bb, save_path):
    """
    Plot the difference between the autoencoder's reconstruction and the original input

    Args:
        input_fts (torch.tensor): the original features of the particles
        reco_fts (torch.tensor): the reconstructed features
        model_fname (str): name of saved model
        bb (str): which black box the input came from
    """
    Path(osp.join(save_path,'reconstruction')).mkdir(exist_ok=True) # make a folder for these graphs
    label = ['$p_x~[GeV]$', '$p_y~[GeV]$', '$p_z~[GeV]$', '$E~[GeV]$']
    feat = ['px', 'py', 'pz' , 'E']

    if model_fname != 'GNN_AE_EdgeConv_Finished':
        loss_type = '$D^{NN}$'
    else:
        loss_type = 'MSE'

    # make a separate plot for each feature
    for i in range(input_fts.shape[1]):
        plt.style.use(hep.style.CMS)
        plt.figure(figsize=(10,8))
        bins = np.linspace(-20, 20, 101)
        if i == 3:  # different bin size for E momentum
            bins = np.linspace(-5, 35, 101)
        plt.ticklabel_format(useMathText=True)
        plt.hist(input_fts[:,i].numpy(), bins=bins, alpha=0.5, label='Input', histtype='step', lw=5)
        plt.hist(reco_fts[:,i].numpy(), bins=bins, alpha=0.5, label='Output', histtype='step', lw=5)
        plt.legend(title='QCD dataset, ' + loss_type, fontsize='x-large')
        plt.xlabel(label[i], fontsize='x-large')
        plt.ylabel('Particles', fontsize='x-large')
        plt.tight_layout()
        plt.savefig(osp.join(save_path, 'reconstruction', feat[i] + '_' + bb + '.pdf'))
        plt.close()

def loss_curves(epochs, early_stop_epoch, train_loss, valid_loss, save_path):
    '''
        Graph our training and validation losses.
    '''
    plt.plot(epochs, train_loss, valid_loss)
    plt.xticks(epochs)
    if early_stop_epoch != None:
        plt.axvline(x=early_stop_epoch, linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['Train', 'Validation', 'Best model'])
    plt.savefig(osp.join(save_path, 'loss_curves.pdf'))
    plt.savefig(osp.join(save_path, 'loss_curves.png'))
    plt.close()
