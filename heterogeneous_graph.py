''' This file converts the given dataset into a heterogeneous graph:
This graph is heterogeneous because there exist different types of nodes and connections. 
'''


import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, HeteroData
import pandas as pd
from sklearn.preprocessing import LabelEncoder


print(f"Torch version: {torch.__version__}")
print(f"Torch geometric version: {torch_geometric.__version__}")


class YelpDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split into raw_dir (downloaded dataset)
        and processed_dir (processed data).
        """
        super(YelpDataset, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered.
        """

        return "yelp_reviews_toy_easier.csv"

    @property
    def processed_file_names(self):
        """If these files are found in raw die, processing is skipped"""

        return "yelp_prepro.pt"

    def download(self):
        pass

    def process(self):
        """
        Our dataset has:
            - Node: User or Business
            - Node_Features: None
            - Edge: Connection between User and Item
            - Edge_Features: Rating of that connection (weight)
        """

        df = pd.read_csv(self.raw_paths[0], header=None)
        df.columns = ['user_id', 'business_id', 'stars']
        print('***********************************************')
        print('Dataset information')
        print(df.nunique())
        print('***********************************************')
        num_users = df['user_id'].nunique()
        num_business = df['business_id'].nunique()

        data = HeteroData()

        # Defining user features (initial embeddings). As we don't have them, we can use the identity matrix
        # Dimension: [num_users, num_features_user]
        data['user'].x = torch.eye(num_users)

        # Defining business features (initial embeddings). As we don't have them, we can use the identity matrix
        # Dimension: [num_business, num_features_business]
        data['business'].x = torch.eye(num_business)

        # Defining the only type of edges that we have
        # Dimension: [2, num_edges_ratings]
        data['user', 'rates', 'business'].edge_index= torch.tensor([list(df.user_id.astype(str).astype(int)), list(df.business_id.astype(str).astype(int))], dtype=torch.long)

        # Defining edge attributes (weights)
        # Dimension: [num_edges_ratings, num_features_edge]
        data['user', 'rates', 'business'].edge_attr = torch.tensor([[rate] for rate in list(df.stars.astype(str).astype(float))])

        torch.save(data, self.processed_paths[0])
