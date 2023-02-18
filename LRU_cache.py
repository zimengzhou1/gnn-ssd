import os.path as osp
import os

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from overflowDataset import OpenFlowDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from neighbor_sampler import NeighborSampler
from functools import lru_cache
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__file__ = os.path.abspath('')
print(__file__)
path = "data"
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)
path = osp.join(osp.realpath(__file__), 'data')
print(path)
dataset = OpenFlowDataset(path)
data = dataset[0]

print(data)

sampleNum = 50000
cacheSize = 200000

print("Initializing sequential neighbor sampler")

def getCacheMiss(sampler, cacheSize):
  @lru_cache(maxsize=cacheSize)
  def get_value(key):
      global cacheMiss
      cacheMiss +=1

  pbar = tqdm(total=sampleNum)
  for step, (batch_size, ids, adjs) in enumerate(sampler):
      for i in ids:
          get_value(i)
      pbar.update(1)
  pbar.close()
  
  return cacheMiss

train_loader = NeighborSampler(data.edge_index, sizes=[10,10], node_idx=data.edge_index[0][:sampleNum])
cacheMiss = 0
print("Cache miss for sequential: ", getCacheMiss(train_loader, cacheSize))


print("Initializing random neighbor sampler")

indices = torch.tensor(random.sample(range(len(data.edge_index[0])), sampleNum))
indices = torch.tensor(indices)
sampled_values = data.edge_index[0][indices]
random_loader = NeighborSampler(data.edge_index, sizes=[10,10], node_idx=sampled_values)
cacheMiss = 0
print("Cache miss for random: ", getCacheMiss(random_loader, cacheSize))



