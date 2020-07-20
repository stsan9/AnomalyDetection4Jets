import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import itertools
import tables
import numpy as np
import pandas as pd
from pyjet import cluster,DTYPE_PTEPM

class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, connect_all=True):
        self._connect_all = connect_all
        super(GraphDataset, self).__init__(root, transform, pre_transform)


    @property
    def raw_file_names(self):
        return ['events_LHCO2020_backgroundMC_Pythia.h5']

    @property
    def processed_file_names(self):
        njets = 2388
        return ['data_{}.pt'.format(i) for i in range(njets)]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass


    def process(self):
        data = []
        for raw_path in self.raw_paths:
            df = pd.read_hdf(raw_path,stop=1000) # just read first 10000 events
            all_events = df.values
            rows = all_events.shape[0]
            cols = all_events.shape[1]
            for i in range(rows):
                pseudojets_input = np.zeros(len([x for x in all_events[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
                for j in range(cols // 3):
                    if (all_events[i][j*3]>0):
                        pseudojets_input[j]['pT'] = all_events[i][j*3]
                        pseudojets_input[j]['eta'] = all_events[i][j*3+1]
                        pseudojets_input[j]['phi'] = all_events[i][j*3+2]
                    pass
                # cluster jets from the particles in one observation
                sequence = cluster(pseudojets_input, R=1.0, p=-1)
                jets = sequence.inclusive_jets()
                for k in range(len(jets)): # for each jet get (px, py, pz, e)
                    if jets[k].pt < 200: continue
                    particles = np.zeros((len(jets[k]),4))
                    for p, part in enumerate(jets[k]):
                        particles[p,:] = np.array([part.px,
                                                   part.py,
                                                   part.pz,
                                                   part.e])
                    data.append(particles)
        ijet = 0
        for d in data:
            nparticles = d.shape[0]
            pairs = [[i, j] for (i, j) in itertools.product(range(nparticles),range(nparticles)) if i!=j]
            edge_index = torch.tensor(pairs, dtype=torch.long)
            edge_index=edge_index.t().contiguous()
            x = torch.tensor(d, dtype=torch.float)
            y = torch.tensor(d, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(ijet)))
            ijet += 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


