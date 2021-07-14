import torch
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

def plot_reco_difference(input_fts, reco_fts, save_path=None):
    """
    Plot the difference between the autoencoder's reconstruction and the original input

    Args:
        input_fts (torch.tensor): the original features of the particles
        reco_fts (torch.tensor): the reconstructed features
    """
    label = ['$p_x~[GeV]$', '$p_y~[GeV]$', '$p_z~[GeV]$']
    feat = ['px', 'py', 'pz']

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
        plt.legend(title='QCD dataset', fontsize='x-large')
        plt.xlabel(label[i], fontsize='x-large')
        plt.ylabel('Particles', fontsize='x-large')
        plt.tight_layout()
        if save_path != None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(osp.join(save_path, feat[i] + '.pdf'))
            plt.close()

def gen_in_out(model, loader, device):
    input_fts = []
    reco_fts = []

    for t in loader:
        model.eval()
        if isinstance(t, list):
            for d in t:
                input_fts.append(d.x)
        else:
            input_fts.append(t.x)
            t.to(device)

        reco_out = model(t)
        if isinstance(reco_out, tuple):
            reco_out = reco_out[0]
        reco_fts.append(reco_out.detach().cpu())

    input_fts = torch.cat(input_fts)
    reco_fts = torch.cat(reco_fts)
    return input_fts, reco_fts

def loss_curves(epochs, early_stop_epoch, train_loss, valid_loss, save_path):
    '''
        Graph our training and validation losses.
    '''
    plt.plot(epochs, train_loss, valid_loss)
    plt.xticks(epochs)
    if epochs[-1] < 100:
        ax = plt.gca()
        ax.locator_params(nbins=10, axis='x')
    if early_stop_epoch != None:
        plt.axvline(x=early_stop_epoch, linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['Train', 'Validation', 'Best model'])
    plt.savefig(osp.join(save_path, 'loss_curves.pdf'))
    plt.savefig(osp.join(save_path, 'loss_curves.png'))
    plt.close()
