import torch
from torch_geometric.datasets import Reddit
import os
import os.path as osp
import numpy as np
import scipy
import json

__file__ = os.path.abspath('')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set environment and path
os.environ['GINEX_NUM_THREADS'] = str(128)

dataset_path = "/mnt/raid0nvme1/zz/ginex/data/Reddit"
dataset = Reddit(dataset_path)

# Construct sparse formats
print('Creating coo/csc/csr format of dataset...')
num_nodes = dataset[0].num_nodes
coo = dataset[0].edge_index.numpy()
v = np.ones_like(coo[0])
coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(num_nodes, num_nodes))
csc = coo.tocsc()
csr = coo.tocsr()
print('Done!')


# Save csc-formatted dataset
indptr = csc.indptr.astype(np.int64)
indices = csc.indices.astype(np.int64)
features = dataset[0].x
labels = dataset[0].y

os.makedirs(dataset_path, exist_ok=True)
indptr_path = os.path.join(dataset_path, 'indptr.dat')
indices_path = os.path.join(dataset_path, 'indices.dat')
features_path = os.path.join(dataset_path, 'features.dat')
labels_path = os.path.join(dataset_path, 'labels.dat')
conf_path = os.path.join(dataset_path, 'conf.json')
split_idx_path = os.path.join(dataset_path, 'split_idx.pth')

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
features_mmap = np.memmap(features_path, mode='w+', shape=dataset[0].x.shape, dtype=np.float32)
features_mmap[:] = features[:]
features_mmap.flush()
print('Done!')

print('Saving labels...')
labels = labels.type(torch.float32)
labels_mmap = np.memmap(labels_path, mode='w+', shape=dataset[0].y.shape, dtype=np.float32)
labels_mmap[:] = labels[:]
labels_mmap.flush()
print('Done!')

print('Making conf file...')
mmap_config = dict()
mmap_config['num_nodes'] = int(dataset[0].num_nodes)
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
mmap_config['labels_shape'] = tuple(labels_mmap.shape)
mmap_config['labels_dtype'] = str(labels_mmap.dtype)
mmap_config['num_classes'] = int(dataset.num_classes)
json.dump(mmap_config, open(conf_path, 'w'))
print('Done!')

print('Saving split index...')
splits = {'train': dataset[0].train_mask, 'test': dataset[0].test_mask, 'valid': dataset[0].val_mask}
torch.save(splits, split_idx_path)
print('Done!')

# Calculate and save score for neighbor cache construction
print('Calculating score for neighbor cache construction...')
score_path = os.path.join(dataset_path, 'nc_score.pth')
out_deg_path = os.path.join(dataset_path, 'out_deg.pth')
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