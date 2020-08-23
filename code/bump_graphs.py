"""
Generate Plots for Bump Hunting
"""
import matplotlib as plt
import torch
import torch.multiprocessing as mp
from EdgeNet import EdgeNet
from graph_data import GraphDataset

cuts = [0.95, 0.97, 0.99, 0.995]  # loss thresholds percentiles

"""
m_12 = sqrt ( (E_1 + E_2)^2 - (p_x1 + p_x2)^2 - (p_y1 + p_y2)^2 - (p_z1 + p_z2)^2 )
"""
def invariant_mass(jet1, jet2):
    return math.sqrt((jet1.e + jet2.e)**2 - (jet1.px + jet2.px)**2 - (jet1.py + jet2.py)**2 - (jet1.pz + jet2.pz)**2)

"""
@params
idx - start idx for indexing into dataset
chunk_size - how much of dataset to process
dataset - graph data
"""
def calc_jet_data(idx, chunk_size, dataset):
    # output tensor colms: e, px, py, pz, loss
    out = torch.zeros(chunk_size, 5)

    # create model and set to evaluation mode
    input_dim = 4
    big_dim = 32
    hidden_dim = 2
    model = EdgeNet(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim)
    modpath = osp.join('/anomalyvol/models/gnn/',model_fname+'.best.pth')
    model.load_state_dict(torch.load(modpath))
    model.eval()
    
    # COME BACK AND EDIT
    for i in range(idx, idx + chunk_size):
        e.put(data[6])
        px.put(data[3])
        py.put(data[4])
        pz.put(data[5])

        try:
            data = data[0]
            batch_output = model(data)
            batch_loss_item = mse(batch_output, data.y)
            losses.put(batch_loss_item)
        except:
            losses.put(0) # model can't evaluate tiny jet

def split_processes(dataset, n_proc):
    chunk_size = len(dataset) // n_proc
    processes = []
    for n in n_proc:
        p = mp.Process(target=calc_jet_data, args=(n*chunk_size, chunk_size, dataset))
        processes.append(p)
    for p in processes:
        p.join()
    
            
def bump_hunt(model_fname, n_proc):
    # specify dataset
    # identify indices for each process
    # loop generating processes and pass in args
    # come back from processes
    # find outliers and dijet im
    # plot and save
    bb1 = GraphDataset('/anomalyvol/data/gnn_geom/bb1/', bb=1)
    
    bb2 = GraphDataset('/anomalyvol/data/gnn_geom/bb2/', bb=2)
    bb3 = GraphDataset('/anomalyvol/data/gnn_geom/bb3/', bb=3)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, help="saved modelname discluding file extension", required=True)
    parser.add_argument("--n-proc", type=int, default=4, help="number of concurrent processes")
    args = parser.parse_args()
        
    gdata = GraphDataset(root=args.dataset, bb=args.bb, n_proc=args.n_proc,
                         n_events=args.n_events, n_particles=args.n_particles)