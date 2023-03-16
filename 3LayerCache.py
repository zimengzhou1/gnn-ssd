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
parser.add_argument('--dataset', type=str, default='reddit')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

metadata = {}
sizes = [int(size) for size in args.sizes.split(',')]

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

# Load dictionary for LFUImproved
with open('/mnt/raid0nvme1/zz/cache_data/' + dataName + ".json", 'r') as f:
  LFUFreqs = json.load(f)

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

# Assume GPU cache is 0.25% of data
GPUCacheNum = int(total_nodes / 200)
node_ids = torch.flatten(orig_edge_index.t())
nodes_to_sample = node_ids[len(node_ids) - subset*2:]
nodes_to_sample_unique_num = torch.unique(nodes_to_sample).numel()
metadata['uniqueNodesInEdges'] = nodes_to_sample_unique_num
metadata['totalEdgesSampled'] = subset

CPUCacheLRU  = LRUCache(CPUCacheNum)
CPUCacheStatic = LRUCache(CPUCacheNum)
CPUCacheLFU = LFUCache(CPUCacheNum)
CPUCacheLFUImproved = LFUCacheImp(CPUCacheNum)
CPUCacheARC = ARCCache(CPUCacheNum)
GPUCache = LRUCache(GPUCacheNum)

# We sample from end of data
#print("Initializing sampler")
# loader = NeighborSampler(data.edge_index, sizes=sizes, node_idx=nodes_to_sample, batch_size=2)
# print(orig_edge_index[0].numel()- subset)
sample_idx = orig_edge_index[0].numel() - subset
sampled_edges = torch.stack([orig_edge_index[0][sample_idx:], orig_edge_index[1][sample_idx:]])
neighbor_loader = LinkNeighborLoader(data, num_neighbors=sizes, edge_label_index=sampled_edges, edge_label_time=data.edge_attr[sample_idx:], time_attr='ts_n')

# Get out neighbor statistics
# Take first 80% of data to find top out-degree neighbors
sample_num = orig_edge_index[0].numel() - int(orig_edge_index[0].numel() / (100/80))
sampled_edges = torch.stack([orig_edge_index[0][:sample_num], orig_edge_index[1][:sample_num]])
# print(sampled_edges)
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
# print(len([i for i in list(sorted_vals_out) if int(i)!=0]))
# print(len(sorted_vals_out))
# print(sorted_vals_out[:20])
# print(sorted_vals_out[len(sorted_vals_out)-10:])

# Populate static cache with highest out degree nodes
for i in range(CPUCacheNum):
    val = int(indices_out[i])
    CPUCacheStatic.put(val, val)

def getHitRate(stats):
    return sum(stats)/len(stats)

# Run LRU and static Cache
def run(CPUCacheLRU, CPUCacheStatic, CPUCacheLFU, CPUCacheARC, CPUCacheLFUImproved):
  numEdgeProcessed = 0
  # pbar = tqdm(total=subset*2)
  pbar = tqdm(total=subset)
  # for batch_size, ids, adjs in loader:
  for data in neighbor_loader:
    # for i in ids:
    
    nodes = data.n_id
    nodes = torch.unique(nodes)
    for i in nodes:
      i = int(i)
      CPUCacheStatic.get(i)
      LRUval = CPUCacheLRU.get(i)
      LFUval = CPUCacheLFU.get(i)
      ARCval = CPUCacheARC.get(i)
      LFUImpval = CPUCacheLFUImproved.get(i)
      if (LFUImpval == -1):
         putVal = CPUCacheLFUImproved.put(i,i, LFUFreqs[str(i)])
      if (LFUval == -1):
         putVal = CPUCacheLFU.put(i,i)
      if (ARCval == -1):
         CPUCacheARC.put(i,i)
      if (LRUval == -1):
        putVal = CPUCacheLRU.put(i,i)
        if putVal == 1:
          metadata['capacityReached'] = numEdgeProcessed
          #print(f"After {numEdgeProcessed} edges ({numEdgeProcessed*100/subset:.2f}%), cache capacity reached")
    numEdgeProcessed += 1
    # pbar.update(batch_size)
    pbar.update(1)
  pbar.close()

  t1 = torch.tensor(CPUCacheLRU.stats)
  #pathStart = '/mnt/raid0nvme1/zz/cache_data/'
  pathStart = './results/'
  commonFilePath = dataName + "_subset_" + str(args.subsetPerc) + "Cache_" + str(args.CPUCachePerc) + "Size_" + args.sizes.replace(",","_")
  #torch.save(t1, pathStart + dataName + "/" + "LRU_" + commonFilePath + '.pt')
  t2 = torch.tensor(CPUCacheStatic.stats)
  #torch.save(t2, pathStart + dataName + "/" + "static_" + commonFilePath + '.pt')
  t3 = torch.tensor(CPUCacheLFU.stats)
  #torch.save(t3, pathStart + dataName + "/" + "LFU_" + commonFilePath + '.pt')
  t4 = torch.tensor(CPUCacheARC.stats)
  #torch.save(t4, pathStart + dataName + "/" + "ARC_" + commonFilePath + '.pt')

  metadata['LRUAccuracy'] = getHitRate(CPUCacheLRU.stats)
  metadata['StaticAccuracy'] = getHitRate(CPUCacheStatic.stats)
  metadata['LFUAccuracy'] = getHitRate(CPUCacheLFU.stats)
  metadata['ARCAccuracy'] = getHitRate(CPUCacheARC.stats)
  metadata['LFUImpAccuracy'] = getHitRate(CPUCacheLFUImproved.stats)
  if 'capacityReached' not in metadata:
     metadata['capacityFurthestReached'] = len(CPUCacheLRU.stats)

  with open(pathStart + dataName + "/" +  "meta_" + commonFilePath + ".json", 'w') as fp:
    json.dump(metadata, fp)

print("Running requests...")
run(CPUCacheLRU, CPUCacheStatic, CPUCacheLFU, CPUCacheARC, CPUCacheLFUImproved)