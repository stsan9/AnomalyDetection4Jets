"""
quick script to check how deepemd performs on identical jets
using 10k events from bb0
"""
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from loss_util import LossFunction, get_ptetaphi
from graph_data import GraphDataset
from torch_geometric.data import DataLoader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, help="batch size", default=1, required=False)
    parser.add_argument("--alt", action='store_true', help="test with normalization", default=True, required=False)
    parser.add_argument("--l2-strength", type=float, help="l2 str", default=1e-8, required=False)
    args = parser.parse_args()
    batch_size = args.batch_size

    # load data
    gdata = GraphDataset(root='/anomalyvol/data/tiny2',bb=0)
    data = []
    for d in gdata: # break down files
        data += d
    device = 'cuda'
    loader = DataLoader(data, batch_size=batch_size, pin_memory=True, shuffle=False)

    if args.alt:
        # normalize
        max_pt, max_eta, max_phi = (0,0,0)
        min_pt, min_eta, min_phi = (9999999,9999999,9999999)
        for d in loader:
            ptetaphi = get_ptetaphi(d.x)

            large_pt = torch.max(ptetaphi[:,0])
            small_pt = torch.min(ptetaphi[:,0])
            if large_pt > max_pt:
                max_pt = large_pt
            if small_pt < min_pt:
                min_pt = small_pt

            large_eta = torch.max(ptetaphi[:,1])
            small_eta = torch.min(ptetaphi[:,1])
            if large_eta > max_eta:
                max_eta = large_eta
            if small_eta < min_eta:
                min_eta = small_eta

            large_phi = torch.max(ptetaphi[:,2])
            small_phi = torch.min(ptetaphi[:,2])
            if large_phi > max_phi:
                max_phi = large_phi
            if small_phi < min_phi:
                min_phi = small_phi

            d.x = ptetaphi

        for d in loader:
            # normalize pt
            d.x[:,0] = (d.x[:,0] - min_pt) / (max_pt - min_pt)
            # normalize eta
            d.x[:,1] = -1 + 2 * (d.x[:,1] - min_eta) / (max_eta - min_eta)
            # normalize phi
            d.x[:,2] = -1 + 2 * (d.x[:,2] - min_phi) / (max_phi - min_phi)

            inds = torch.LongTensor([1,2,0])
            d.x = torch.index_select(d.x, 1, inds)

    if not args.alt:
        deepemd = LossFunction('deep_emd_loss', device=device)
    else:
        deepemd = LossFunction('deep_emd_loss_alt', device=device)

    # calculate emds
    losses = []
    t = tqdm.tqdm(loader,total=len(data)/batch_size)
    for b in t:
        if len(b.x) > 30:
            continue
        b.to(device)
        loss = deepemd.loss_ftn(b.x, b.x, b.batch, l2_strength=args.l2_strength)
        losses += loss.tolist()
        t.refresh()

    losses = np.array(losses)
    np.save(f'/anomalyvol/info/deepemdlosses_l2_{args.l2_strength}_normalized_{args.alt}', losses)

    # analysis
    max_emd = np.around(max(losses), decimals=3)
    mu = np.format_float_scientific(np.mean(losses), precision=3)
    sigma = np.format_float_scientific(np.std(losses), precision=3)

    # plot
    plt.figure(figsize=(7,5.8))
    hts,bins,_=plt.hist(losses,bins=100)
    plt.xlabel("EMD", fontsize=16)
    x = max(bins) * 0.6
    y = max(hts) * 0.8
    plt.text(x, y, f'$\mu={mu}$'
                '\n'
                f'$\sigma={sigma}$'
                '\n'
                f'$max={max_emd}$')
    plt.savefig(f'/anomalyvol/info/emd_losses_l2_{args.l2_strength}_normalized_{args.alt}.png')
