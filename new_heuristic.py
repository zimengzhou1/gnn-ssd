import os.path as osp
import os
import sys
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
import argparse

# Set arguments


# Datset to use
dataName = 'wiki' # 'overflow', 'taobao' , 'reddit', wiki'
parser = argparse.ArgumentParser('Cache testing')
parser.add_argument('--sizes', type=str, default='10,10')
parser.add_argument('--dataset', type=str, default='reddit')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

dataName = args.dataset
print("using datset: ", dataName)
sizes = [int(size) for size in args.sizes.split(',')]
print("size: ", sizes)

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

n1 = torch.unique(orig_edge_index[0])
n2 = torch.unique(orig_edge_index[1])
total_nodes = torch.unique(torch.cat((n1,n2))).numel()

#loader = NeighborSampler(data.edge_index, sizes=[10,10], node_idx=nodes_to_sample, batch_size=2)
loader = NeighborSampler(data.edge_index, sizes=sizes, node_idx=torch.unique(torch.cat((n1,n2))), batch_size=1)

sample_cnt = {}
cnt = 0
pbar = tqdm(total=total_nodes)
for batch_size, ids, adjs in loader:
    sample_cnt[cnt] = len(ids)
    cnt +=1
    pbar.update(batch_size)
pbar.close()

sample_cnt = {k: v for k, v in sorted(sample_cnt.items(), key=lambda item: item[1], reverse=True)}

import json
path = '/mnt/raid0nvme1/zz/cache_data/'
with open(path + dataName + "_new" + ".json", 'w') as fp:
    json.dump(sample_cnt, fp)