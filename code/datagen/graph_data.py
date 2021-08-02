"""
    Custom PyTorch dataloader. Use command "python graph_data.py --h"
    or look at the main function for details on how to run. Handles
    preprocessing collision event data into particle-level data
    represented using graphs in PyTorch geometric. If processed data
    is already present in specified directory, will just load in
    data.
"""
import glob
import torch
import tables
import itertools
import numpy as np
import pandas as pd
import os.path as osp
import multiprocessing
from pathlib import Path
from pyjet import cluster,DTYPE_PTEPM
from torch_geometric.data import Dataset, Data

from util.gdata_util import jet_particles

def process_func(args):
    self, raw_path, k = args
    return self.process_one_chunk(raw_path, k)

# functions needed from original pytorch dataset class for overwriting _process ###
def to_list(x):
    if not isinstance(x, (tuple, list)) or isinstance(x, str):
        x = [x]
    return x

# augmented to be less robust but faster than original (remove check for all files)
def files_exist(files):
    return len(files) != 0

def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

def collate(items): # collate function for data loaders (transforms list of lists to list)
    l = sum(items, [])
    return Batch.from_data_list(l)

class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 bb=0, n_events=-1, n_proc=1, R=1.0, n_events_merge=100):
        """
        Initialize parameters of graph dataset
        Args:
            root (str): path
            n_particles (int): particles + padding for jet (default -1=no padding)
            bb (int): dataset to read in (0=background)
            n_events (int): how many events to process (-1=all)
            n_proc (int): number of processes to split into
            n_events_merge (int): how many events to merge
        """
        max_events = int(1.1e6 if bb == -1 else 1e6)
        self.n_particles = n_particles
        self.bb = bb
        self.n_events = max_events if n_events==-1 else n_events
        self.n_events_merge = n_events_merge
        self.n_proc = n_proc
        self.R = R
        self.chunk_size = self.n_events // self.n_proc
        self.file_string = ['data_{}.pt', 'data_bb1_{}.pt', 'data_bb2_{}.pt', 'data_bb3_{}.pt', 'data_rnd_{}.pt']
        super(GraphDataset, self).__init__(root, transform, pre_transform)


    @property
    def raw_file_names(self):
        """
        Determines which file is being processed
        """
        files = [['events_LHCO2020_backgroundMC_Pythia.h5'],
                 ['events_LHCO2020_BlackBox1.h5'],
                 ['events_LHCO2020_BlackBox2.h5'],
                 ['events_LHCO2020_BlackBox3.h5'],
                 ['events_anomalydetection.h5']]
        return files[self.bb]

    @property
    def processed_file_names(self):
        """
        Returns a list of all the files in the processed files directory
        """
        proc_list = glob.glob(osp.join(self.processed_dir, 'data*.pt'))
        return_list = list(map(osp.basename, proc_list))
        return return_list

    def len(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process_one_chunk(self, raw_path, k):
        """
        Handles conversion of dataset file at raw_path into graph dataset.

        Args:
            raw_path (str): The absolute path to the dataset file
            k (int): Counter of the process used to separate chunk of data to process
        """
        df = pd.read_hdf(raw_path, start = k * self.chunk_size, stop = (k + 1) * self.chunk_size)
        part_gen = jet_particles(df, R=self.R, u=True)

        datas = []
        for particles, n_particles, jet_mass, jet_px, jet_py, jet_pz, jet_e, signal_bit, row in part_gen:

            if self.n_particles != -1:   # 0 pad / fix length
                if self.n_particles < n_particles:
                    particles = particles[:self.n_particles]
                else:
                    z = np.zeros((self.n_particles,3))
                    z[:particles.shape[0],:] = particles
                    particles = z

            event_idx = k*self.chunk_size + row
            pairs = np.stack([[m, n] for (m, n) in itertools.product(range(n_particles),range(n_particles)) if m!=n])
            edge_index = torch.tensor(pairs, dtype=torch.long)
            edge_index=edge_index.t().contiguous()
            x = torch.tensor(particles, dtype=torch.float) # node attribute and AE target
            u = torch.tensor([event_idx, n_particles, jet_mass, jet_px, jet_py, jet_pz, jet_e, signal_bit], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            data.u = torch.unsqueeze(u, 0)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            datas.append([data])

            if row % self.n_events_merge == self.n_events_merge-1:
                datas = sum(datas,[])
                torch.save(datas, osp.join(self.processed_dir, self.file_string[self.bb].format(event_idx)))
                datas = []

        if len(data) != 0:  # save any extras
            datas = sum(datas,[])
            torch.save(datas, osp.join(self.processed_dir, self.file_string[self.bb].format(event_idx)))

    def process(self):
        """
        Split processing of dataset across multiple processes.
        """
        for raw_path in self.raw_paths:
            pars = []
            for k in range(self.n_events // self.chunk_size):
                # to do it sequentially
                #self.process_one_chunk(raw_path, k)
                # to do it with multiprocessing
                pars += [(self, raw_path, k)]
            pool = multiprocessing.Pool(self.n_proc)
            pool.map(process_func, pars)

    def get(self, idx):
        """ Used by PyTorch DataSet class """
        p = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(p)
        return data
    
    def _process(self):
        """
        Checks if we want to process the raw file into a dataset. If files 
        already present skips processing.
        """
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != __repr__(self.pre_transform):
            logging.warning(
                'The `pre_transform` argument differs from the one used in '
                'the pre-processed version of this dataset. If you really '
                'want to make use of another pre-processing technique, make '
                'sure to delete `{}` first.'.format(self.processed_dir))
        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != __repr__(self.pre_filter):
            logging.warning(
                'The `pre_filter` argument differs from the one used in the '
                'pre-processed version of this dataset. If you really want to '
                'make use of another pre-fitering technique, make sure to '
                'delete `{}` first.'.format(self.processed_dir))

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        print('Processing...')

        Path(self.processed_dir).mkdir(exist_ok=True)
        self.process()

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        torch.save(__repr__(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        torch.save(__repr__(self.pre_filter), path)

        print('Done!')
