"""
Generate Plots for Bump Hunting
"""
import matplotlib as plt
import torch
import torch.multiprocessing as mp
import os.path as osp
from EdgeNet import EdgeNet
from graph_data import GraphDataset

cuts = [0.95, 0.97, 0.99, 0.995]  # loss thresholds percentiles
model_fname = ""

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
out - output tensor with columns: [e, px, py, pz, loss]
"""
def calc_jet_data(idx, chunk_size, dataset, out, model):

    # create model and set to evaluation mode

    with torch.no_grad():
        for i in range(idx, idx + chunk_size):
            out[i][0] = dataset[i][6] # e
            out[i][1] = dataset[i][3] # px
            out[i][2] = dataset[i][4] # py
            out[i][3] = dataset[i][5] # pz

            try:
                data = dataset[i][0]
                batch_output = model(data)
                batch_loss_item = mse(batch_output, data.y)
                out[i][4] = batch_loss_item
            except:
                out[i][4] = 0 # model can't evaluate tiny jet

def split_processes(dataset, n_proc):
    print("Splitting...")
    chunk_size = len(dataset) // n_proc
    processes = []
    out = torch.empty(len(dataset), 5, dtype=torch.float32)
    out.share_memory_()
    
    # define model for loss calculations
    input_dim = 4
    big_dim = 32
    hidden_dim = 2
    model = EdgeNet(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim)
    modpath = osp.join('/anomalyvol/models/gnn/',model_fname+'.best.pth')
    model.load_state_dict(torch.load(modpath))
    model.eval()
    model.share_memory()
    
    dataset.share_memory()

    # generate processes
    for n in range(n_proc):
        print("Making process: " + str(n))
        if n == n_proc - 1: # account for rounding error of chunk_size
            chunk_size = len(dataset) - n * chunk_size
        p = mp.Process(target=calc_jet_data, args=[n*chunk_size, chunk_size, dataset, out, model])
        p.start()
        processes.append(p)
    # end processes
    for p in processes:
        print("Ending process")
        p.join()
    
    print(out[-5])
    
            
def bump_hunt(n_proc):
    # specify dataset
    # identify indices for each process
    # loop generating processes and pass in args
    # come back from processes
    # find outliers and dijet im
    # plot and save
    print("loading in bb1...")
    bb1 = GraphDataset('/anomalyvol/data/gnn_geom/bb1', bb=1)
    print("done loading")
    split_processes(bb1, n_proc)
    
    #bb2 = GraphDataset('/anomalyvol/data/gnn_geom/bb2/', bb=2)
    #bb3 = GraphDataset('/anomalyvol/data/gnn_geom/bb3/', bb=3)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, help="saved modelname discluding file extension", required=True)
    parser.add_argument("--n-proc", type=int, default=4, help="number of concurrent processes")
    args = parser.parse_args()
    
    model_fname = args.modelname
    bump_hunt(args.n_proc)