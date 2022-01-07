# This file converts the given dataset into a bipartite graph pytorch can work with

import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
from sklearn.preprocessing import LabelEncoder


print(f"Torch version: {torch.__version__}")
print(f"Torch geometric version: {torch_geometric.__version__}")


class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class YelpDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. THis folder is split into raw_dir (downloaded dataset)
        and processed_dir (processed data).
        """
        super(YelpDataset, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered.
        """

        return "yelp_reviews_toy.csv"

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
        print('Dataset information')
        print(df.nunique())

        # We need to encode user and business ids in order to make a correct graph
        entity_encoder = LabelEncoder()
        users = set(list(df['user_id']))
        businesses = set(list(df['business_id']))

        # We are assuming that user_id and business_id are two different sets with no intersection
        if list(users.intersection(businesses)) != []:
            print('AN ID OF A USER MATCHES AN ID FROM A BUSINESS!!!')
            print(users.intersection(businesses))
            return

        # Encode labels
        entities = list(users.union(businesses))
        entity_encoder.fit(entities)
        df['business_id'] = entity_encoder.transform(df.business_id)
        df['user_id'] = entity_encoder.transform(df.user_id)

        # Definition of edges and weights
        edge_attr = torch.tensor([[rate] for rate in list(df.stars)])

        user_id = (torch.Tensor(df['user_id'].values)).long()
        business_id = (torch.Tensor(df['business_id'].values)).long()
        c, cidx = torch.unique(input=user_id, return_inverse=True)
        v, vidx = torch.unique(input=business_id, return_inverse=True)
        edge_index = torch.tensor([list(df.user_id), list(df.business_id)], dtype=torch.long)
        x_s = cidx.unique()
        x_t = vidx.unique()
        data = BipartiteData(edge_index, x_s=x_s, x_t=x_t)
        data.edge_attr = edge_attr
        data.num_users = len(x_s)
        data.num_business = len(x_t)
        torch.save(data, self.processed_paths[0])
