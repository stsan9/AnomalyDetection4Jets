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

# load data
gdata = GraphDataset(root='/anomalyvol/data/tiny2',bb=0)
data = []
for d in gdata: # break down files
    data += d
device = 'cuda:0'
loader = DataLoader(data, batch_size=64, pin_memory=True, shuffle=False)
deepemd = LossFunction('deep_emd_loss', device=device)

# calculate emds
losses = []
t = tqdm.tqdm(loader,total=len(data)/64)
for b in t:
    b.to(device)
    loss = deepemd.loss_ftn(b.x, b.x, b.batch) # reformats data before feeding into emd_loss
    losses += loss.tolist()

losses = np.array(losses)
np.save('/anomalyvol/info/deepemdlosses', losses)

# analysis
max_emd = np.around(max(losses), decimals=3)
mu = np.format_float_scientific(np.mean(losses), precision=3)
sigma = np.format_float_scientific(np.std(losses), precision=3)

# plot
plt.figure(figsize=(7,5.8))
hts,bins,_=plt.hist(losses,bins=100)
plt.xlabel("EMD", fontsize=16)
x = min(bins)
y = max(hts) * 0.8
plt.text(x, y, f'$\mu={mu}$'
               '\n'
               f'$\sigma={sigma}$'
               '\n'
               f'$max={max_emd}$')
plt.savefig('/anomalyvol/info/emd_losses.png')