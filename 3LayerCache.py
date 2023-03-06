from collections import OrderedDict
import os.path as osp
import os
import sys
import torch
from overflowDataset import OverFlowDataset
from torch_geometric.datasets import JODIEDataset
from tqdm import tqdm
from neighbor_sampler import NeighborSampler
import argparse
import scipy
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import json


parser = argparse.ArgumentParser('Cache testing')
parser.add_argument('--subsetPerc', type=float, default=1, help='percentage of data to sample from')
parser.add_argument('--CPUCachePerc', type=int, default=20, help='CPU cache percentage of data')
# parser.add_argument('--LRUOnly', default=True, action='store_true')
# parser.add_argument('--no-LRUOnly', dest='LRUOnly', action='store_false')
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
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'overflow')
    dataset = OverFlowDataset(path)
    data = dataset[0]
    orig_edge_index = data.edge_index
elif dataName == 'taobao':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'taobao', 'taobao.pt')
    data = torch.load(path)
    orig_edge_index = data.edge_index
    data.edge_index = to_undirected(data.edge_index)
elif dataName == 'reddit':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
    dataset = JODIEDataset(path, name='reddit')
    data_orig = dataset[0]
    data = Data(x=data_orig.msg, edge_index=torch.stack([data_orig.src, data_orig.dst], dim=0), edge_attr=data_orig.t)
    orig_edge_index = data.edge_index
    data.edge_index = to_undirected(data.edge_index)
elif dataName == 'wiki':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
    dataset = JODIEDataset(path, name='wikipedia')
    data_orig = dataset[0]
    data = Data(x=data_orig.msg, edge_index=torch.stack([data_orig.src, data_orig.dst], dim=0), edge_attr=data_orig.t)
    orig_edge_index = data.edge_index
    data.edge_index = to_undirected(data.edge_index)
 
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.stats = []
        self.capacityReached = False
 
    def get(self, key: int) -> int:
        if key not in self.cache:
            self.stats.append(0)
            return -1
        else:
            self.stats.append(1)
            self.cache.move_to_end(key)
            return self.cache[key]
 
    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)
            if not self.capacityReached:
                self.capacityReached = True
                return 1
        return 0

subset = int(orig_edge_index[0].numel() / (100/args.subsetPerc))

n1 = torch.unique(orig_edge_index[0])
n2 = torch.unique(orig_edge_index[1])
total_nodes = torch.unique(torch.cat((n1,n2))).numel()

# Assume CPU cache is 20% of data
CPUCacheNum = int(total_nodes / (100/args.CPUCachePerc))
metadata['cacheSize'] = CPUCacheNum

# Assume GPU cache is 0.25% of data
GPUCacheNum = int(total_nodes / 200)
node_ids = torch.flatten(data.edge_index.t())
nodes_to_sample = node_ids[len(node_ids) - subset*2:]
nodes_to_sample_unique_num = torch.unique(nodes_to_sample).numel()
metadata['uniqueNodesInEdges'] = nodes_to_sample_unique_num
metadata['totalEdgesSampled'] = subset
# print("Total number of unique nodes in dataset: ", total_nodes)
# print("Number of unique nodes in edges we sample: ", nodes_to_sample_unique_num)
# print("Number of total edges sampled: ", subset)


CPUCacheLRU  = LRUCache(CPUCacheNum)
CPUCacheStatic = LRUCache(CPUCacheNum)
GPUCache = LRUCache(GPUCacheNum)

# We sample from end of data
#print("Initializing sampler")
loader = NeighborSampler(data.edge_index, sizes=sizes, node_idx=nodes_to_sample, batch_size=2)

# Get out neighbor statistics
coo = orig_edge_index.numpy()
v = np.ones_like(coo[0])
coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(orig_edge_index[0].numel(), orig_edge_index[0].numel()))
csc = coo.tocsc()
csr = coo.tocsr()

csc_indptr_tensor = torch.from_numpy(csc.indptr.astype(np.int64))
csr_indptr_tensor = torch.from_numpy(csr.indptr.astype(np.int64))
out_num_neighbors = csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]
in_num_neighbors = (csc_indptr_tensor[1:] - csc_indptr_tensor[:-1])
sorted_vals_out, indices_out = torch.sort(out_num_neighbors, descending=True)
sorted_vals_in, indices_in = torch.sort(in_num_neighbors, descending=True)

# Populate static cache with highest out degree nodes
for i in range(CPUCacheNum):
    val = int(indices_out[i])
    CPUCacheStatic.put(val, val)

def getHitRate(stats):
    return sum(stats)/len(stats)

# Run LRU and static Cache
def run(CPUCacheLRU, CPUCacheStatic):
  numEdgeProcessed = 0
  #pbar = tqdm(total=subset*2)
  for batch_size, ids, adjs in loader:
    for i in ids:
      i = int(i)
      val = CPUCacheLRU.get(i)
      CPUCacheStatic.get(i)
      if (val == -1):
        # Fetch from SSD
        putVal = CPUCacheLRU.put(i,i)
        if putVal == 1:
          metadata['capacityReached'] = numEdgeProcessed
          #print(f"After {numEdgeProcessed} edges ({numEdgeProcessed*100/subset:.2f}%), cache capacity reached")
    numEdgeProcessed += 1
    #pbar.update(batch_size)
  #pbar.close()

  t1 = torch.tensor(CPUCacheLRU.stats)
  commonFilePath = dataName + "_subset_" + str(args.subsetPerc) + "Cache_" + str(args.CPUCachePerc) + "Size_" + args.sizes.replace(",","_")
  torch.save(t1, "cache_data/" + dataName + "/" + "LRU_" + commonFilePath + '.pt')
  t2 = torch.tensor(CPUCacheStatic.stats)
  torch.save(t2, "cache_data/" + dataName + "/" + "static_" + commonFilePath + '.pt')

  metadata['LRUAccuracy'] = getHitRate(CPUCacheLRU.stats)
  metadata['StaticAccuracy'] = getHitRate(CPUCacheStatic.stats)
  if 'capacityReached' not in metadata:
     metadata['capacityFurthestReached'] = len(CPUCacheLRU.stats)

  with open("cache_data/" + dataName + "/" +  "meta_" + commonFilePath + ".json", 'w') as fp:
    json.dump(metadata, fp)

print("Running requests...")
run(CPUCacheLRU, CPUCacheStatic)