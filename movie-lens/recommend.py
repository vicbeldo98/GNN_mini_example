import os.path as osp
import torch
from torch.nn import Linear
import torch.nn.functional as F

import torch_geometric.transforms as T
from dataset import MovieLens
from train import Model

MODEL_PATH = osp.join(osp.dirname(osp.realpath(__file__)), 'model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
dataset = MovieLens(path, model_name='all-MiniLM-L6-v2')
data = dataset[0].to(device)

# Add user node features for message passing:
data['user'].x = torch.eye(data['user'].num_nodes, device=device)
del data['user'].num_nodes

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.'''

model = Model(hidden_channels=32).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
pred = model(data.x_dict, data.edge_index_dict, data['user', 'movie'].edge_label_index)
pred = pred.clamp(min=0, max=5)
print(pred)