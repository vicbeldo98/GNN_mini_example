# This file load the dataset and aims to create a model for recommendation (NOT FINISHED): 
# DOCUMENTATION: https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
from heterogeneous_graph import YelpDataset
import torch

dataset = YelpDataset(root="data/")
data = dataset[0]

'''Automatically convert a homogenous GNN model to a heterogeneous GNN model by making use of torch_geometric.nn.to_hetero() 
or torch_geometric.nn.to_hetero_with_bases()'''

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GNN(hidden_channels=64, out_channels=8)
model = to_hetero(model, data.metadata(), aggr='sum')
