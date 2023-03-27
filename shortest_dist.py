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
from torch_geometric.utils import to_networkx
import networkx as nx

# Set arguments
# Percentage of data to sample from
subsetPerc = 10

# CPU cache percentage of nodes
CPUCachePerc = 100

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

print("creating networkx...")
G = to_networkx(data)


subsetPerc = 1
num_to_sample = int(len(orig_edge_index[0]) * (subsetPerc/100))

print("sampled number: ", num_to_sample)

new_edges = torch.stack([orig_edge_index[0][-num_to_sample:], orig_edge_index[1][-num_to_sample:]])
pairs = new_edges.t()
j = 0
tot = 0
no_paths = 0

pbar = tqdm(total=len(pairs)-1)
for i in range(len(pairs)-1):
  fst_pair = pairs[i]
  snd_pair = pairs[i+1]
  try:
    
    n1=nx.shortest_path_length(G, source=int(fst_pair[0]), target=int(snd_pair[0]))
    n2=nx.shortest_path_length(G, source=int(fst_pair[0]), target=int(snd_pair[1]))
    n3=nx.shortest_path_length(G, source=int(fst_pair[1]), target=int(snd_pair[0]))
    n4=nx.shortest_path_length(G, source=int(fst_pair[1]), target=int(snd_pair[1]))
    tot += (n1+n2+n3+n4)/4
  except nx.NetworkXNoPath:
      no_paths += 1
  pbar.update(1)
pbar.close()


print(tot/(num_to_sample-no_paths))
print(no_paths)
