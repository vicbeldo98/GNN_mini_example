import os.path as osp
import torch
from torch.nn import Linear
import torch.nn.functional as F

import torch_geometric.transforms as T
from dataset import MovieLens
from train import Model

MODEL_PATH = osp.join(osp.dirname(osp.realpath(__file__)), 'models/model_10000')

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

import argparse
import getpass

parser = argparse.ArgumentParser()
parser.add_argument('-u', metavar="user", default='1', help="UserId for which we want to make a recommendation")
parser.add_argument('-n', metavar="movies", default='1', help="Number of movies we want to recommend")
parsed_args = parser.parse_args()
args = vars(parsed_args)

USERID = int(args['u'])
NUM_MOVIES = int(args['n'])

# Loading data
import pandas as pd
df_movies = pd.read_csv('data/raw/ml-latest-small/movies.csv', index_col='movieId')
df_ratings = pd.read_csv('data/raw/ml-latest-small/ratings.csv')

movie_mapping = {i: idx for i, idx in enumerate(df_movies.index)}
num_movies = len(data['movie'].x)

row = torch.tensor([USERID] * num_movies)
col = torch.arange(num_movies)
edge_label_index = torch.stack([row, col], dim=0)
pred = model(data.x_dict, data.edge_index_dict, edge_label_index)
pred = pred.clamp(min=0, max=5)
idx_max = torch.topk(pred, NUM_MOVIES).indices
print('Recommended movies for userId ' + str(USERID))
for i in idx_max:
    movieId = movie_mapping[int(i)]
    print(df_movies.loc[movieId].title)