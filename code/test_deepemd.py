"""
quick script to check how deepemd performs on identical jets
using 10k events from bb0
"""
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from loss_util import LossFunction
from graph_data import GraphDataset
from torch_geometric.data import DataLoader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, help="batch size", default=1, required=False)
    parser.add_argument("--l2-strength", type=float, help="l2 str", default=1e-8, required=False)
    args = parser.parse_args()
    batch_size = args.batch_size

    # load data
    gdata = GraphDataset(root='/anomalyvol/data/tiny2',bb=0)
    data = []
    for d in gdata: # break down files
        data += d
    device = 'cuda:0'
    loader = DataLoader(data[:5000], batch_size=batch_size, pin_memory=True, shuffle=False)

    deepemd = LossFunction('deep_emd_loss', device=device)

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
    np.save(f'/anomalyvol/info/deepemdlosses_l2_{args.l2_strength}_normalized', losses)

    # analysis
    max_emd = np.around(max(losses), decimals=3)
    min_emd = np.around(min(losses), decimals=3)
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
                f'$max={max_emd}$'
                '\n'
                f'$min={min_emd}$')
    plt.savefig(f'/anomalyvol/info/emd_losses_l2_{args.l2_strength}_normalized.png')
