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
    args = parser.parse_args()
    batch_size = args.batch_size

    # load data
    gdata = GraphDataset(root='/anomalyvol/data/tiny2',bb=0)
    data = []
    for d in gdata: # break down files
        data += d
    device = 'cuda'
    loader = DataLoader(data, batch_size=batch_size, pin_memory=True, shuffle=False)
    deepemd = LossFunction('deep_emd_loss', device=device)

    # calculate emds
    losses = []
    t = tqdm.tqdm(loader,total=len(data)/batch_size)
    for b in t:
        if len(b.x) > 30:
            continue
        b.to(device)
        loss = deepemd.loss_ftn(b.x, b.x, b.batch, l2_strength=1e-8) # reformats data before feeding into emd_loss
        losses += loss.tolist()
        t.refresh()

    losses = np.array(losses)
    np.save('/anomalyvol/info/deepemdlosses_l2_1e-8', losses)

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
    plt.savefig('/anomalyvol/info/emd_losses_l2_1-e8.png')
