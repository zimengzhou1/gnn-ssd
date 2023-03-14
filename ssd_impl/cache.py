import torch
from lib.utils import *
import threading
from queue import Queue
import numpy as np
import json
import os
from tqdm import tqdm

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

    # The address table of the cache has num_nodes entries each of which is a single
    # int32 value. This can support the cache with up to 2147483647 entries.
    table_size = 4 * self.num_nodes
    self.num_entries = int((self.size-table_size)/4/self.feature_dim)
    if self.num_entries > torch.iinfo(torch.int32).max:
      raise ValueError
    
  # Fill cache with the feature vectors corresponding to the given indices. It is called
  # when initializing the feature cache in order to reduce the cold misses.
  def fill_cache(self, indices):
    self.address_table = torch.full((self.num_nodes,), -1, dtype=torch.int32)
    self.address_table[indices] = torch.arange(indices.numel(), dtype=torch.int32)
    orig_num_threads = torch.get_num_threads() 
    torch.set_num_threads(int(os.environ['GINEX_NUM_THREADS']))
    self.cache = self.mmapped_features[indices].cpu()
    torch.set_num_threads(orig_num_threads)

  # Assume cache is LRU
