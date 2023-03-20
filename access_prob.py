from collections import OrderedDict, defaultdict
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
from sklearn import preprocessing


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
 
class LRUCache:
    def __init__(self):
        self.counts = {}
 
    def get(self, key: int) -> int:
        if key not in self.counts:
            self.counts[key] = 0
        else:
            self.counts[key] += 1
 

subset = int(orig_edge_index[0].numel() / (100/args.subsetPerc))

n1 = torch.unique(orig_edge_index[0])
n2 = torch.unique(orig_edge_index[1])
total_nodes = torch.unique(torch.cat((n1,n2))).numel()

# Assume CPU cache is 20% of data
CPUCacheNum = int(total_nodes / (100/args.CPUCachePerc))
metadata['cacheSize'] = CPUCacheNum

node_ids = torch.flatten(orig_edge_index.t())
nodes_to_sample = node_ids[len(node_ids) - subset*2:]
nodes_to_sample_unique_num = torch.unique(nodes_to_sample).numel()
metadata['uniqueNodesInEdges'] = nodes_to_sample_unique_num
metadata['totalEdgesSampled'] = subset

CPUCacheLRU  = LRUCache()

# We sample from end of data
loader = NeighborSampler(data.edge_index, sizes=sizes, node_idx=nodes_to_sample, batch_size=2)

# Get out neighbor statistics
coo = data.edge_index.numpy()
print(coo.shape)
print(total_nodes)
v = np.ones_like(coo[0])
coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(total_nodes,total_nodes))
csc = coo.tocsc()
csr = coo.tocsr()

csc_indptr_tensor = torch.from_numpy(csc.indptr.astype(np.int64))
csr_indptr_tensor = torch.from_numpy(csr.indptr.astype(np.int64))
out_num_neighbors = csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]
in_num_neighbors = (csc_indptr_tensor[1:] - csc_indptr_tensor[:-1])
sorted_vals_out, indices_out = torch.sort(out_num_neighbors, descending=True)
sorted_vals_in, indices_in = torch.sort(in_num_neighbors, descending=True)
# Here idx_out gives a mapping of node_val -> ranking
vals, idx_out = torch.sort(indices_out)

def calculateCoef(l1, l2):
  #assert len(l1) == len(l2)
  #assert len(l2) == total_nodes
  difSum = 0
   
  # for i in range(len(l1)):
  #   difSum += (l1.index(i) - l2.index(i))**2

  l2Reordered = [l2[i] for i in l1]

  le = preprocessing.LabelEncoder()
  le.fit(l2Reordered)
  l2 = list(le.transform(l2Reordered))
  
  print(l2Reordered[:5])
  print(l2[:5])
  print(l2Reordered == l2)

  for i in range(len(l1)):
    difSum += (i - l2[i])**2
    
  return 1-(6*difSum/(len(l1) * (len(l1)**2 -1)))
   

# Run LRU and static Cache
def run(CPUCacheLRU):
  pbar = tqdm(total=subset*2)
  for batch_size, ids, adjs in loader:
    for i in ids:
      CPUCacheLRU.get(int(i))
    pbar.update(batch_size)
  pbar.close()

  x = CPUCacheLRU.counts
  sorted_x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}
  sampledList = list(sorted_x.keys())
  outDegList = idx_out.tolist()

  print("sampled size: ", len(sampledList))
  print("total size: ", len(outDegList))

  coef = calculateCoef(sampledList, outDegList)

  print("Size: ", sizes, " coef of: ", coef)


  # t1 = torch.tensor(CPUCacheLRU.stats)
  # commonFilePath = dataName + "_subset_" + str(args.subsetPerc) + "Cache_" + str(args.CPUCachePerc) + "Size_" + args.sizes.replace(",","_")
  # torch.save(t1, "cache_data/" + dataName + "/" + "LRU_" + commonFilePath + '.pt')
  # t2 = torch.tensor(CPUCacheStatic.stats)
  # torch.save(t2, "cache_data/" + dataName + "/" + "static_" + commonFilePath + '.pt')
  # t3 = torch.tensor(CPUCacheLFU.stats)
  # torch.save(t3, "cache_data/" + dataName + "/" + "LFU_" + commonFilePath + '.pt')

  # metadata['LRUAccuracy'] = getHitRate(CPUCacheLRU.stats)
  # metadata['StaticAccuracy'] = getHitRate(CPUCacheStatic.stats)
  # metadata['LFUAccuracy'] = getHitRate(CPUCacheLFU.stats)
  # if 'capacityReached' not in metadata:
  #    metadata['capacityFurthestReached'] = len(CPUCacheLRU.stats)

  # with open("cache_data/" + dataName + "/" +  "meta_" + commonFilePath + ".json", 'w') as fp:
  #   json.dump(metadata, fp)

print("Running requests...")
run(CPUCacheLRU)