from cacheImpl import *
import os.path as osp
import os
import sys
import torch
from overflowDataset import OverFlowDataset
from torch_geometric.datasets import JODIEDataset
from tqdm import tqdm
from neighbor_sampler import NeighborSampler
from torch_geometric.loader import LinkNeighborLoader
import argparse
import scipy
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import json


parser = argparse.ArgumentParser('Cache testing')
parser.add_argument('--subsetPerc', type=float, default=1, help='percentage of data to sample from')
parser.add_argument('--CPUCachePerc', type=int, default=20, help='CPU cache percentage of data')
parser.add_argument('--sizes', type=str, default='10,10')
parser.add_argument('--staticStart', type=int, default=0)
parser.add_argument('--staticEnd', type=int, default=80)
parser.add_argument('--dataset', type=str, default='reddit')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

metadata = {}
sizes = [int(size) for size in args.sizes.split(',')]

print("static start: ", args.staticStart)
print("static end: ", args.staticEnd)

__file__ = os.path.abspath('')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'OpenFlow')

dataName = args.dataset
print("Using ", args.subsetPerc, "% of data, CPU cache of: ", args.CPUCachePerc, "% dataset: ", dataName)

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

subset = int(orig_edge_index[0].numel() / (100/args.subsetPerc))

n1 = torch.unique(orig_edge_index[0])
n2 = torch.unique(orig_edge_index[1])
total_nodes = torch.unique(torch.cat((n1,n2))).numel()

data.n_id = torch.arange(total_nodes)
data.ts_n = torch.zeros(total_nodes).to(dtype=torch.long)
data.x = data.n_id

# Assume CPU cache is 20% of data
CPUCacheNum = int(total_nodes / (100/args.CPUCachePerc))
metadata['cacheSize'] = CPUCacheNum

node_ids = torch.flatten(orig_edge_index.t())
nodes_to_sample = node_ids[len(node_ids) - subset*2:]
nodes_to_sample_unique_num = torch.unique(nodes_to_sample).numel()
metadata['uniqueNodesInEdges'] = nodes_to_sample_unique_num
metadata['totalEdgesSampled'] = subset

CPUCacheStatic = LRUCache(CPUCacheNum)
CPUCacheStaticClo = LRUCache(CPUCacheNum)
CPUCacheStaticBet = LRUCache(CPUCacheNum)
CPUCacheStaticEig = LRUCache(CPUCacheNum)

# We sample from end of data
#print("Initializing sampler")
# loader = NeighborSampler(data.edge_index, sizes=sizes, node_idx=nodes_to_sample, batch_size=2)
# print(orig_edge_index[0].numel()- subset)
sample_idx = orig_edge_index[0].numel() - subset
sampled_edges = torch.stack([orig_edge_index[0][sample_idx:], orig_edge_index[1][sample_idx:]])
neighbor_loader = LinkNeighborLoader(data, num_neighbors=sizes, edge_label_index=sampled_edges, edge_label_time=data.edge_attr[sample_idx:], time_attr='ts_n')

# Get out neighbor statistics
# Take first 80% of data to find top out-degree neighbors
if args.staticStart == 0:
   sample_num_start = 0
else:
  sample_num_start = int(orig_edge_index[0].numel() / (100/args.staticStart))
sample_num_end = int(orig_edge_index[0].numel() / (100/args.staticEnd))
#print("num of edges used to construct ranking: ", sample_num_end - sample_num_start)
metadata['static_edges_used'] = sample_num_end - sample_num_start
#print("how many edges until start of sample: ", sample_idx - sample_num_end)
metadata['edge_til_start'] = sample_idx - sample_num_end
sampled_edges = torch.stack([orig_edge_index[0][sample_num_start:sample_num_end], orig_edge_index[1][sample_num_start:sample_num_end]])
# print(sampled_edges)
if (dataName != 'overflow'):
  sampled_edges = to_undirected(sampled_edges)
# Find number of unique nodes in 80% of data
#n1 = torch.unique(sampled_edges[0])
#n2 = torch.unique(sampled_edges[1])
#total_nodes = torch.unique(torch.cat((n1,n2))).numel()

coo = sampled_edges.numpy()
#coo = data.edge_index.numpy()
v = np.ones_like(coo[0])
coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(total_nodes, total_nodes))
csc = coo.tocsc()
csr = coo.tocsr()

csc_indptr_tensor = torch.from_numpy(csc.indptr.astype(np.int64))
csr_indptr_tensor = torch.from_numpy(csr.indptr.astype(np.int64))
out_num_neighbors = csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]
in_num_neighbors = (csc_indptr_tensor[1:] - csc_indptr_tensor[:-1])
sorted_vals_out, indices_out = torch.sort(out_num_neighbors, descending=True)
sorted_vals_in, indices_in = torch.sort(in_num_neighbors, descending=True)

path = '/mnt/raid0nvme1/zz/'

filename_bet = path + dataName + "_between.json"
filename_clo = path + dataName + "_close.json"
filename_eig = path + dataName + "_eigen.json"

with open(filename_bet) as json_file:
    data_bet = json.load(json_file)
with open(filename_clo) as json_file:
    data_clo = json.load(json_file)
with open(filename_eig) as json_file:
    data_eig = json.load(json_file)

data_bet = dict(sorted(data_bet.items(), key=lambda item: item[1], reverse=True))
data_clo = dict(sorted(data_clo.items(), key=lambda item: item[1], reverse=True))
data_eig = dict(sorted(data_eig.items(), key=lambda item: item[1], reverse=True))

indices_out_bet = [float(i) for i in list(data_bet.keys())]
indices_out_clo = [float(i) for i in list(data_clo.keys())]
indices_out_eig = [float(i) for i in list(data_eig.keys())]


# Populate static cache with highest out degree nodes
for i in range(CPUCacheNum):
  if (i >= len(indices_out_bet)):
     break
  val = int(indices_out[i])
  CPUCacheStatic.put(val, val)
  
  val_bet = int(indices_out_bet[i])
  CPUCacheStaticBet.put(val_bet, val_bet)

  val_clo = int(indices_out_clo[i])
  CPUCacheStaticClo.put(val_clo, val_clo)

  val_eig = int(indices_out_eig[i])
  CPUCacheStaticEig.put(val_eig, val_eig)

def getHitRate(stats):
    return sum(stats)/len(stats)

# Run LRU and static Cache
def run(CPUCacheStatic, CPUCacheStaticBet, CPUCacheStaticClo, CPUCacheStaticEig):
  # pbar = tqdm(total=subset*2)
  pbar = tqdm(total=subset)
  # for batch_size, ids, adjs in loader:
  for data in neighbor_loader:   
    nodes = data.n_id
    nodes = torch.unique(nodes)
    for i in nodes:
      i = int(i)
      CPUCacheStatic.get(i)
      CPUCacheStaticBet.get(i)
      CPUCacheStaticClo.get(i)
      CPUCacheStaticEig.get(i)
    # pbar.update(batch_size)
    pbar.update(1)
  pbar.close()

  print("static: ", getHitRate(CPUCacheStatic.stats))
  metadata['static_hit'] = getHitRate(CPUCacheStatic.stats)
  metadata['clo_hit'] = getHitRate(CPUCacheStaticClo.stats)
  metadata['bet_hit'] = getHitRate(CPUCacheStaticBet.stats)
  metadata['eig_hit'] = getHitRate(CPUCacheStaticEig.stats)

  pathStart = './results/'
  commonFilePath = "centrality_" + str(args.CPUCachePerc)
  with open(pathStart + dataName + "/" +  "meta_" + commonFilePath + ".json", 'w') as fp:
    json.dump(metadata, fp)

print("Running requests...")
run(CPUCacheStatic, CPUCacheStaticBet, CPUCacheStaticClo, CPUCacheStaticEig)