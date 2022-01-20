
## Trying to accomplish an example of GNN for recommendation systems in Pytorch Geometric

# MOVIELENS DATASET

This dataset (ml-latest-small) describes 5-star rating from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.

Users were selected at random for inclusion. All selected users had rated at least 20 movies. No demographic information is included. Each user is represented by an id, and no other information is provided.

The data are contained in the files `movies.csv` and `ratings.csv`. More details about the contents and use of all these files follows.

This is a *development* dataset. As such, it may change over time and is not an appropriate dataset for shared research results. See available *benchmark* datasets if that is your intent.

This and other GroupLens data sets are publicly available for download at <http://grouplens.org/datasets/>.


### ENVIRONMENT OF WORK
conda create -n my-torch python=3.8 -y

conda activate my-torch

conda install pip

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

~/anaconda3/envs/my-torch/bin/pip install torch-geometric==2.0.3

~/anaconda3/envs/my-torch/bin/pip install torch-sparse

~/anaconda3/envs/my-torch/bin/pip install torch-scatter

conda install -c conda-forge huggingface_hub==0.2.1

conda install -c conda-forge sentence-transformers

### SIMILAR ISSUES FOUND:

https://github.com/pyg-team/pytorch_geometric/issues/1999

## EXECUTE SCRIPT

### Train the recommendation model

python3 train.py

### Recommend k movies to specific userId

python3 recommend_k_movies.py -u 10 -n 6 (-u (userId), -n (number of movies to recommend))
