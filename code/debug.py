import torch
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
import models
import emd_models
from loss_util import LossFunction
from graph_data import GraphDataset
from plot_util import loss_curves
from torch_geometric.nn import DataParallel
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = getattr(models, 'EdgeNetEMD')(input_dim=3, big_dim=32, hidden_dim=2)
model.load_state_dict(torch.load('/anomalyvol/debug/debug_model.pth'))
model = DataParallel(model)
model.to(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

optimizer.zero_grad()
gdata = torch.load('/anomalyvol/debug/debug_input.pt')[0]
loader = DataListLoader(gdata, batch_size=12,pin_memory=True, shuffle=False)

for b in loader:
    model(b)
print('done')
