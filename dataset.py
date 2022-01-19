# Code from: https://github.com/pyg-team/pytorch_geometric/blob/8943288ed3f7b70ef6621c4e8b9ec58af3ae24de/torch_geometric/datasets/movie_lens.py#L12
from typing import Optional, Callable, List

import os
import os.path as osp

import torch

from torch_geometric.data import (InMemoryDataset, HeteroData, download_url,
                                  extract_zip)


class MovieLens(InMemoryDataset):
    r"""A heterogeneous rating dataset, assembled by GroupLens Research from
    the `MovieLens web site <https://movielens.org>`_, consisting of nodes of
    type :obj:`"movie"` and :obj:`"user"`.
    User ratings for movies are available as ground truth labels for the edges
    between the users and the movies :obj:`("user", "rates", "movie")`.
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        model_name (str): Name of model used to transform movie titles to node
            features. The model comes from the`Huggingface SentenceTransformer
            <https://huggingface.co/sentence-transformers>`_.
    """

    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

    def __init__(self, root, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 model_name: Optional[str] = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join('ml-latest-small', 'movies.csv'),
            osp.join('ml-latest-small', 'ratings.csv'),
        ]

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.model_name}.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        '''
        self.raw_paths[0] -> movies.csv:  movieId (int) | title (str)  | genres (str)
        self.raw_paths[1] -> ratings.csv: userId (int)  | movieId (int)| rating (float) | timestamp

        '''
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        data = HeteroData()

        df = pd.read_csv(self.raw_paths[0], index_col='movieId')

        ################# MOVIE NODES #################
        # We create a dict to map movieId with numbers
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        # Get a vector representing which genres each movie belong (https://pandas.pydata.org/docs/reference/api/pandas.Series.str.get_dummies.html#pandas.Series.str.get_dummies)
        genres = df['genres'].str.get_dummies('|').values
        genres = torch.from_numpy(genres).to(torch.float)

        # SentenceTransformer: This framework generates embeddings for each input sentence (https://pypi.org/project/sentence-transformers/)
        model = SentenceTransformer(self.model_name)
        with torch.no_grad():
            emb = model.encode(df['title'].values, show_progress_bar=True,
                               convert_to_tensor=True).cpu()

        # Concatenates embedding extracted from title and genres, and puts it as features of movie nodes (https://pytorch.org/docs/stable/generated/torch.cat.html)
        data['movie'].x = torch.cat([emb, genres], dim=-1)

        ################# USER NODES #################
        df = pd.read_csv(self.raw_paths[1])
        user_mapping = {idx: i for i, idx in enumerate(df['userId'].unique())}
        data['user'].num_nodes = len(user_mapping)

        ################# EDGES #################
        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = torch.tensor([src, dst])

        rating = torch.from_numpy(df['rating'].values).to(torch.long)
        data['user', 'rates', 'movie'].edge_index = edge_index
        data['user', 'rates', 'movie'].edge_label = rating

        torch.save(self.collate([data]), self.processed_paths[0])