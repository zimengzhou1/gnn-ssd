import os.path as osp
import os
import torch
from overflowDataset import OverFlowDataset
from torch_geometric.datasets import JODIEDataset
from tqdm import tqdm
from neighbor_sampler import NeighborSampler
import scipy
import numpy as np
from collections import OrderedDict
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from cacheImpl import *
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.loader import LinkNeighborLoader
import json

# Set arguments
# Percentage of data to sample from
subsetPerc = 3

# CPU cache percentage of nodes
CPUCachePerc = 20

# Datset to use
dataName = 'taobao' # 'overflow', 'taobao' , 'reddit', wiki'

# Load data
__file__ = os.path.abspath('')

print("Loading dataset ", dataName, "...")

if dataName == 'overflow':
    path = '/mnt/raid0nvme1/zz/data/' + 'overflow'
    dataset = OverFlowDataset(path)
    data = dataset[0]
    orig_edge_index = data.edge_index
elif dataName == 'taobao':
    path = '/mnt/raid0nvme1/zz/data/' + 'taobao/taobao.pt'
    data = torch.load(path)
    orig_edge_index = data.edge_index
    data.edge_index = to_undirected(data.edge_index)
elif dataName == 'reddit':
    path = '/mnt/raid0nvme1/zz/data/' + 'JODIE'
    dataset = JODIEDataset(path, name='reddit')
    data_orig = dataset[0]
    data = Data(x=data_orig.msg, edge_index=torch.stack([data_orig.src, data_orig.dst], dim=0), edge_attr=data_orig.t)
    orig_edge_index = data.edge_index
    data.edge_index = to_undirected(data.edge_index)
elif dataName == 'wiki':
    path = '/mnt/raid0nvme1/zz/data/' + 'JODIE'
    dataset = JODIEDataset(path, name='wikipedia')
    data_orig = dataset[0]
    data = Data(x=data_orig.msg, edge_index=torch.stack([data_orig.src, data_orig.dst], dim=0), edge_attr=data_orig.t)
    orig_edge_index = data.edge_index
    data.edge_index = to_undirected(data.edge_index)

print(data)


# Take first 80% of data to find top out-degree neighbors
sample_num = int(orig_edge_index[0].numel() / (100/84))
sampled_edges = torch.stack([orig_edge_index[0][:sample_num], orig_edge_index[1][:sample_num]])
# print(sampled_edges)
if (dataName != 'overflow'):
  sampled_edges = to_undirected(sampled_edges)

n1 = torch.unique(sampled_edges[0])
n2 = torch.unique(sampled_edges[1])
total_nodes = torch.unique(torch.cat((n1,n2)))

data = Data(x=total_nodes, edge_index=sampled_edges)

print("creating networkx...")
G = to_networkx(data)

print("closeness done!")
eigen = nx.eigenvector_centrality_numpy(G)

path = '/mnt/raid0nvme1/zz/'
with open(path + dataName + "_eigen" + ".json", 'w') as fp:
    json.dump(eigen, fp)


