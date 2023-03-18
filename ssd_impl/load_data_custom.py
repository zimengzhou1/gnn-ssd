
import os
import os.path as osp
import sys
import inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
from torch_geometric.datasets import Reddit
from torch_geometric.utils import to_undirected
from overflowDataset import OverFlowDataset
from torch_geometric.datasets import JODIEDataset
import numpy as np
import scipy
import json
from torch_geometric.data import Data

__file__ = os.path.abspath('')
dataName = 'taobao'

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
    n1 = torch.unique(orig_edge_index[0])
    n2 = torch.unique(orig_edge_index[1])
    total_nodes = torch.unique(torch.cat((n1,n2))).numel()
    x = torch.rand(total_nodes, 128).to(torch.long)
    data.x = x
    path = '/mnt/raid0nvme1/zz/data/' + 'taobao'
elif dataName == 'reddit':
    path = '/mnt/raid0nvme1/zz/data/' + 'JODIE'
    dataset = JODIEDataset(path, name='reddit')
    data_orig = dataset[0]
    data = Data(x=data_orig.msg, edge_index=torch.stack([data_orig.src, data_orig.dst], dim=0), edge_attr=data_orig.t)
    orig_edge_index = data.edge_index
    data.edge_index = to_undirected(data.edge_index)
    path = path + "/reddit"
elif dataName == 'wiki':
    path = '/mnt/raid0nvme1/zz/data/' + 'JODIE'
    dataset = JODIEDataset(path, name='wikipedia')
    data_orig = dataset[0]
    data = Data(x=data_orig.msg, edge_index=torch.stack([data_orig.src, data_orig.dst], dim=0), edge_attr=data_orig.t)
    orig_edge_index = data.edge_index
    data.edge_index = to_undirected(data.edge_index)
    path = path + "/wikipedia"

# Construct sparse formats
print('Creating coo/csc/csr format of dataset...')
num_nodes = total_nodes
print(data)
coo = data.edge_index.numpy()
v = np.ones_like(coo[0])
coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(num_nodes, num_nodes))
csc = coo.tocsc()
csr = coo.tocsr()
print('Done!')

# Save csc-formatted dataset
indptr = csc.indptr.astype(np.int64)
indices = csc.indices.astype(np.int64)
features = data.x

#os.makedirs(path, exist_ok=True)
indptr_path = os.path.join(path, 'indptr.dat')
indices_path = os.path.join(path, 'indices.dat')
features_path = os.path.join(path, 'features.dat')
sampled_path = os.path.join(path, 'sampled.dat')
conf_path = os.path.join(path, 'conf.json')

print('Saving indptr...')
indptr_mmap = np.memmap(indptr_path, mode='w+', shape=indptr.shape, dtype=indptr.dtype)
indptr_mmap[:] = indptr[:]
indptr_mmap.flush()
print('Done!')

print('Saving indices...')
indices_mmap = np.memmap(indices_path, mode='w+', shape=indices.shape, dtype=indices.dtype)
indices_mmap[:] = indices[:]
indices_mmap.flush()
print('Done!')

print('Saving features...')
features_mmap = np.memmap(features_path, mode='w+', shape=data.x.shape, dtype=np.float32)
features_mmap[:] = features[:]
features_mmap.flush()
print('Done!')

print('Making conf file...')
mmap_config = dict()
mmap_config['num_nodes'] = int(total_nodes)
mmap_config['indptr_shape'] = tuple(indptr.shape)
mmap_config['indptr_dtype'] = str(indptr.dtype)
mmap_config['indices_shape'] = tuple(indices.shape)
mmap_config['indices_dtype'] = str(indices.dtype)
mmap_config['indices_shape'] = tuple(indices.shape)
mmap_config['indices_dtype'] = str(indices.dtype)
mmap_config['indices_shape'] = tuple(indices.shape)
mmap_config['indices_dtype'] = str(indices.dtype)
mmap_config['features_shape'] = tuple(features_mmap.shape)
mmap_config['features_dtype'] = str(features_mmap.dtype)
json.dump(mmap_config, open(conf_path, 'w'))
print('Done!')

print('Saving sampled edges')
# We take last 10% of edges
subset = int(orig_edge_index[0].numel() / (100/10))
sample_idx = orig_edge_index[0].numel() - subset
sampled_edges = torch.stack([orig_edge_index[0][sample_idx:], orig_edge_index[1][sample_idx:]])
torch.save(sampled_edges, sampled_path)
print('Done!')


# Calculate and save score for neighbor cache construction
print('Calculating score for neighbor cache construction...')
score_path = os.path.join(path, 'nc_score.pth')
out_deg_path = os.path.join(path, 'out_deg.pth')
csc_indptr_tensor = torch.from_numpy(csc.indptr.astype(np.int64))
csr_indptr_tensor = torch.from_numpy(csr.indptr.astype(np.int64))

eps = 0.00000001
in_num_neighbors = (csc_indptr_tensor[1:] - csc_indptr_tensor[:-1]) + eps
out_num_neighbors = (csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]) + eps
sorted_vals_out, indices_out = torch.sort(out_num_neighbors, descending=True)
print(sorted_vals_out[:10])
print(indices_out[:10])
score = out_num_neighbors / in_num_neighbors
print('Done!')

print('Saving score...')
torch.save(score, score_path)
torch.save(indices_out, out_deg_path)
print('Done!')