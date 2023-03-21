import torch
from lib.utils import *
import threading
from queue import Queue
import numpy as np
import json
import os
from tqdm import tqdm
from collections import OrderedDict, defaultdict

class FeatureCache:
  '''
  Args:
    size (int): the size of the feature cache including both the cached data and 
      the address table in byte.
    num_nodes (int): the number of nodes in the graph.
    mmapped_features (Tensor): the tensor memory-mapped to the feature vectors
    feature_dim (int): the dimension of the feature vectors
    exp_name (str): the name of the experiments used to designate the path of the
      runtime trace files.
    verbose (bool): if set, the detailed processing information is displayed
  '''

  def __init__(self, size, num_nodes, mmapped_features, feature_dim, verbose):
    self.size = size
    self.num_nodes = num_nodes
    self.mmapped_features = mmapped_features
    self.feature_dim = feature_dim
    self.verbose = verbose
    self.cache = OrderedDict()
    
  # Fill cache with the feature vectors corresponding to the given indices. It is called
  # when initializing the feature cache in order to reduce the cold misses.
  def fill_cache(self, indices, useLRU):
    self.address_table = torch.full((self.num_nodes,), -1, dtype=torch.int32)
    self.address_table[indices] = torch.arange(indices.numel(), dtype=torch.int32)
    orig_num_threads = torch.get_num_threads() 
    torch.set_num_threads(int(os.environ['GINEX_NUM_THREADS']))
    if (not useLRU):
      self.cache = self.mmapped_features[indices].cpu()
    else:
      for i in indices:
        i = int(i)
        self.cache[i] = self.mmapped_features[i].cpu()
        self.cache.move_to_end(i)
    torch.set_num_threads(orig_num_threads)

  # Assume cache is LRU

  def lru_gather(self, indices):
    #out = torch.empty((indices.numel(), self.feature_dim), dtype=torch.int32)
    out = []
    cnt = 0
    hits = 0
    misses = 0

    for i in indices:
      i = int(i)
      if (not torch.is_tensor(self.get(i))):
        misses +=1
        val = self.mmapped_features[i]
        #out[cnt] = val
        out.append(val)
        self.put(i, val)
      else:
        hits += 1
        out.append(self.cache[i])
        #out[cnt] = self.cache[i]
    
    return out, hits, misses

  def get(self, key: int) -> int:
    if key not in self.cache:
        return -1
    else:
        self.cache.move_to_end(key)
        return self.cache[key]
 
  def put(self, key: int, value: int) -> int:
    self.cache[key] = value
    self.cache.move_to_end(key)
    if len(self.cache) > self.size:
        self.cache.popitem(last = False)