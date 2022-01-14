# This file load the dataset and aims to create a model for recommendation (NOT FINISHED).
from dataset import YelpDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling, GatedGraphConv, SAGEConv, SGConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score
import numpy as np


dataset = YelpDataset(root="data/")
batch_size = 256
embed_dim = 128
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

'''model = Sequential('x, edge_index', [
    (GCNConv(in_channels, 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (GCNConv(64, 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    Linear(64, out_channels),
])'''

for data in loader:
    edge_index_T = torch.stack([data.edge_index[1], data.edge_index[0]], dim=0)  # Transposed/Reversed graph.

    num_customers = data.num_users
    num_vendors = data.num_business
    data.customer = torch.identity(num_customers)
    data.vendor = torch.identity(num_vendors)
    conv1 = SAGEConv((num_customers, num_vendors), 64)
    new_vendor_x = conv1((data.customer, data.vendor), data.edge_index).relu()
    conv2 = SAGEConv((num_vendors, num_customers), 64)
    new_customer_x = conv2((data.vendor, data.customer), edge_index_T).relu()

    # Repeat with new_vendor_x and new_customer_x:
    conv3 = SAGEConv((64, 64), 128)
    new_vendor_x2 = conv3((new_customer_x, new_vendor_x), data.edge_index).relu()
