"""
Plot how emd model predicts emd values of gae output vs. actual emd values
"""
import glob
import tqdm
import torch
import awkward
import numpy as np
import os.path as osp
import energyflow as ef
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.data import Data, DataListLoader, Batch

import models
from loss_util import LossFunction
from graph_data import GraphDataset

ONE_HUNDRED_GEV = 100.0

def calc_emd(rec, jet):
    rec = rec.numpy()
    jet = jet.numpy()
    # center jet according to pt-centroid
    yphi_avg_rec = np.average(rec[:,1:3], weights=rec[:,0], axis=0)
    rec[:,1:3] -= yphi_avg_rec
    yphi_avg_jet = np.average(jet[:,1:3], weights=jet[:,0], axis=0)
    jet[:,1:3] -= yphi_avg_jet
    emdval = ef.emd.emd(rec, jet) / ONE_HUNDRED_GEV
    return emdval

if __name__ == "__main__":
    saved_models = [osp.basename(x)[:-9] for x in glob.glob('/anomalyvol/models/*')]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, help="Saved model name without file extension", required=True, choices=saved_models)
    parser.add_argument("--model", choices=models.model_list, help="model selection", required=True)
    parser.add_argument("--box-num", type=int, help="0=QCD-background; 1=bb1; 2=bb2; 4=rnd", required=True)
    parser.add_argument("--output-dir", type=str, help="Output directory for files.", required=True, default='/anomalyvol/figures/')
    parser.add_argument("--overwrite", action='store_true', help="Toggle overwrite of saved emds.", default=False, required=False)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True) # make a folder for the graphs of this model   
    true_emds_file = osp.join(args.output_dir,'emds.npy')
    pred_emds_file = osp.join(args.output_dir,'emds.npy')

    if not (osp.isfile(true_emds_file) and osp.isfile(pred_emds_file)) or args.overwrite:
        print("Loading gae")
        # load model
        input_dim = 3
        latent_dim = 2
        model = getattr(models, args.model)(input_dim=input_dim, hidden_dim=latent_dim)
        modpath = osp.join('/anomalyvol/models/',args.model_name+'.best.pth')
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(modpath, map_location=torch.device('cuda')))
        else:
            model.load_state_dict(torch.load(modpath, map_location=torch.device('cpu')))
        emd_nn = LossFunction("emd_loss")
        print(f"Loaded {modpath}")
        
        print("Loading data")
        # load data
        bb_name = ["bb0", "bb1", "bb2", "bb3", "rnd"][args.box_num]
        gdata = GraphDataset('/anomalyvol/data/lead_2/%s/'%bb_name, bb=args.box_num)
        # gdata = GraphDataset('/anomalyvol/data/lead_2/tiny', bb=args.box_num)
        data_loader = DataListLoader(gdata)
        print(f"Loaded {bb_name}")
        
        emds = []
        preds = []
        model.eval()
        # reconstruct jet with model, calculate the true emd with the og jet, compare with emd network
        with torch.no_grad():
            for k, data in tqdm.tqdm(enumerate(data_loader),total=len(data_loader)):
                print(f"{k+1}/{len(data_loader)}")
                data = data[0]  # remove extra bracket from DataListLoader since batch size is 1
                # mask 3rd jet in 3-jet events
                event_list = torch.stack([d.u[0][0] for d in data]).cpu().numpy()
                unique, inverse, counts = np.unique(event_list, return_inverse=True, return_counts=True)
                awk_array = awkward.JaggedArray.fromparents(inverse, event_list)
                mask = ((awk_array.localindex < 2).flatten()) * (counts[inverse]>1)
                data = [d for d,m in zip(data, mask) if m]
                # get leading 2 jets
                data_batch = Batch.from_data_list(data)
                data_batch.x = data_batch.x[:,4:-1]
                jets_x = data_batch.x
                batch = data_batch.batch
                # run inference on all jets
                jets_rec = model(data_batch)
                # for each jet calc true emd and emd network output
                for ib in torch.unique(batch):
                    x = jets_rec[batch==ib]
                    y = jets_x[batch==ib]
                    true_emd = calc_emd(x, y)
                    pred_emd = emd_nn.emd_loss(x, y, torch.tensor(0).repeat(x.shape[0])).item()
                    emds.append(true_emd)
                    preds.append(pred_emd)
        print("Saving values for graphing")
        emds = np.array(emds)
        preds = np.array(preds)
        np.save(true_emds_file, emds)
        np.save(pred_emds_file, preds)
    else:
        print("Loading values for graphing")
        emds = np.load(true_emds_file)
        preds = np.load(pred_emds_file)

    print("Generating graphs")
    max_range = round(np.max(preds),-2)
    min_range = round(np.min(preds),-2)
    # plot overlaying hists 
    fig, ax = plt.subplots(figsize =(5, 5)) 
    plt.hist(emds, bins=np.linspace(min_range, max_range , 101),label='True', alpha=0.5)
    plt.hist(preds, bins=np.linspace(min_range, max_range, 101),label = 'Pred.', alpha=0.5)
    plt.legend()
    ax.set_xlabel('EMD [GeV]') 
    fig.savefig(osp.join(args.output_dir,'overlay_EMD.pdf'))
    fig.savefig(osp.join(args.output_dir,'overlay_EMD.png'))

    # plot 2d hist
    fig, ax = plt.subplots(figsize =(5, 5)) 
    x_bins = np.linspace(min_range, max_range, 101)
    y_bins = np.linspace(min_range, max_range, 101)
    plt.hist2d(emds, preds, bins=[x_bins,y_bins])
    ax.set_xlabel('True EMD [GeV]')  
    ax.set_ylabel('Pred. EMD [GeV]')
    fig.savefig(osp.join(args.output_dir,'EMD_corr.pdf'))
    fig.savefig(osp.join(args.output_dir,'EMD_corr.png'))
