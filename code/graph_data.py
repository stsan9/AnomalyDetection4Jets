"""
    Custom PyTorch dataloader. Use command "python graph_data.py --h"
    or look at the main function for details on how to run. Handles
    preprocessing collision event data into particle-level data
    represented using graphs in PyTorch geometric. If processed data
    is already present in specified directory, will just load in
    data.
"""
import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import itertools
import tables
import numpy as np
import pandas as pd
from pyjet import cluster,DTYPE_PTEPM
import glob
import multiprocessing
from pathlib import Path

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


class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 n_particles=-1, bb=0, n_events=-1, n_proc=1,
                 n_events_merge=100, leading_pair_only = 0, signal_only=False,
                 background_only=False):
        """
        Initialize parameters of graph dataset
        Args:
            root (str): path
            n_particles (int): particles + padding for jet (default -1=no padding)
            bb (int): dataset to read in (0=background)
            n_events (int): how many events to process (-1=all)
            n_proc (int): number of processes to split into
            n_events_merge (int): how many events to merge
            leading_pair_only (int): toggle to process only leading 2 jets / event
        """
        self.n_particles = n_particles
        self.bb = bb
        self.n_events = 1000000 if n_events==-1 else n_events
        self.n_events_merge = n_events_merge
        self.n_proc = n_proc
        self.chunk_size = self.n_events // self.n_proc
        self.file_string = ['data_{}.pt', 'data_bb1_{}.pt', 'data_bb2_{}.pt', 'data_bb3_{}.pt', 'data_rnd_{}.pt']
        self.leading_pair_only = leading_pair_only
        self.signal_only = signal_only
        self.background_only = background_only
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

    def __len__(self):
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
        if self.signal_only:
            df = df.loc[df.iloc[:,-1] == 1] # get only signals [rnd]
        else if self.background_only:
            df = df.loc[df.iloc[:,-1] == 0] # get only background [rnd]
        all_events = df.values
        rows = all_events.shape[0]
        cols = all_events.shape[1]
        datas = []

        for i in range(rows):
            if i%self.n_events_merge == 0:
                datas = []
            event_idx = k*self.chunk_size + i
            if event_idx == self.signal_only:
                event_idx += 2000000 # avoid conflicting event indices when combining signal-only set with others
            ijet = 0
            # cluster jets from the particles in one event
            pseudojets_input = np.zeros(len([x for x in all_events[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
            for j in range(cols // 3):
                if (all_events[i][j*3]>0):
                    pseudojets_input[j]['pT'] = all_events[i][j*3]
                    pseudojets_input[j]['eta'] = all_events[i][j*3+1]
                    pseudojets_input[j]['phi'] = all_events[i][j*3+2]
            sequence = cluster(pseudojets_input, R=1.0, p=-1)
            jets = sequence.inclusive_jets()
            jet_num = 0
            for jet in jets: # for each jet get (px, py, pz, e)
                # if setting true, only get leading 2 jets and skip events with less than 2 jets
                if self.leading_pair_only:
                    if len(jets) < 2:
                        break
                    elif jet_num > 1:
                        break
                if jet.pt < 200 or len(jet)<=1: continue
                if self.n_particles > -1:
                    n_particles = self.n_particles
                else:
                    n_particles = len(jet)
                particles = np.zeros((n_particles, 8))

                # store all the particles of this jet
                for p, part in enumerate(jet):
                    if n_particles > -1 and p >= n_particles: break
                    # save two representations: px, py, pz, e, and pt, eta, phi, mass
                    particles[p,:] = np.array([part.px,
                                               part.py,
                                               part.pz,
                                               part.e,
                                               part.pt,
                                               part.eta,
                                               part.phi,
                                               part.mass])
                if self.n_particles>-1:
                    n_particles = min(len(jet),self.n_particles)
                else:
                    n_particles = len(jet)
                pairs = np.stack([[m, n] for (m, n) in itertools.product(range(n_particles),range(n_particles)) if m!=n])
                # save [deta, dphi] as edge attributes (may not be used depending on model)
                #eta0s = particles[pairs[:,0],6]
                #eta1s = particles[pairs[:,1],6]
                #phi0s = particles[pairs[:,0],7]
                #phi1s = particles[pairs[:,1],7]
                #detas = np.abs(eta0s - eta1s)
                #dphis = (phi0s - phi1s + np.pi) % (2 * np.pi) - np.pi
                #edge_attr = np.stack([detas,dphis],axis=1)
                #edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                signal_bit = all_events[i][-1]
                edge_index = torch.tensor(pairs, dtype=torch.long)
                edge_index=edge_index.t().contiguous()
                # save [px, py, pz, e] of particles as node attributes and target
                x = torch.tensor(particles[:,:4], dtype=torch.float)
                # save [n_particles, mass, px, py, pz, e] of the jet as global attributes
                # (may not be used depending on model)
                u = torch.tensor([event_idx, n_particles, jet.mass, jet.px, jet.py, jet.pz, jet.e, signal_bit], dtype=torch.float)
                data = Data(x=x, edge_index=edge_index)
                data.u = torch.unsqueeze(u, 0)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                datas.append([data])
                ijet += 1

            # save data in separate files for every n_events_merge events
            if i%self.n_events_merge == self.n_events_merge-1:
                datas = sum(datas,[])
                # save data in format (particle_data, event_of_jet, mass_of_jet, px, py, pz, e, signal_bit)
                torch.save(datas, osp.join(self.processed_dir, self.file_string[self.bb].format(event_idx)))
        
        if i%self.n_events_merge != self.n_events_merge-1 and len(data) != 0: # save any lefovers
            datas = sum(datas,[])
            torch.save(datas, osp.join(self.processed_dir, self.file_string[self.bb].format(event_idx)))

    def process(self):
        """
        Split processing of dataset across multiple processes.
        """
        print(len(self.processed_file_names))
        # only do 10000 events for background, process full blackboxes
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset path", required=True)
    parser.add_argument("--n-proc", type=int, default=1, help="number of concurrent processes")
    parser.add_argument("--n-events", type=int, default=-1, help="number of events (-1 means all)")
    parser.add_argument("--n-particles", type=int, default=-1, help="max number of particles per jet with zero-padding (-1 means all)")
    parser.add_argument("--bb", type=int, default=0, help="black box number (0 is background, -1 is the mixed rnd set)")
    parser.add_argument("--n-events-merge", type=int, default=100, help="number of events to merge")
    parser.add_argument("--leading-pair-only", type=int, default=0, help="if we only want the leading 2 jets of each event (0: False, not 0: True)")
    parser.add_argument("--signal-only", action='store_true', help="Proc events with signal only [rnd]", default=False, required=False)
    parser.add_argument("--background-only", action='store_true', help="Proc events with background only [rnd]", default=False, required=False)
    args = parser.parse_args()

    if args.bb != -1 and (args.signal_only or args.background_only):
        exit("Can only use --signal-only/--background-only flag with RND dataset")

    gdata = GraphDataset(root=args.dataset, bb=args.bb, n_proc=args.n_proc,
                         n_events=args.n_events, n_particles=args.n_particles,
                         n_events_merge=args.n_events_merge, leading_pair_only=args.leading_pair_only,
                         signal_only=args.signal_only, background_only=args.background_only)
