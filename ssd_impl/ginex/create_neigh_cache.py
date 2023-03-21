import argparse
import os

from lib.data import *
from lib.cache import *


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
argparser.add_argument('--neigh-cache-size', type=int, default=4500000000)
argparser.add_argument('--ginex-num-threads', type=int, default=128)
args = argparser.parse_args()

# Set environment and path
os.environ['GINEX_NUM_THREADS'] = str(args.ginex_num_threads)
# dataset_path = os.path.join('./dataset', args.dataset + '-ginex')
dataset_path = "../taobao"
score_path = os.path.join(dataset_path, 'nc_score.pth')
# split_idx_path = os.path.join(dataset_path, 'split_idx.pth')


def save_neighbor_cache():
    print('Creating neighbor cache...')
    # dataset = GinexDataset(path=dataset_path, split_idx_path=split_idx_path, score_path=score_path)
    score_path = dataset_path + "/nc_score.pth"
    score = torch.load(score_path)
    # score = dataset.get_score()

    conf_path = os.path.join(dataset_path, 'conf.json')
    conf = json.load(open(conf_path, 'r'))

    # rowptr, col = dataset.get_adj_mat()
    indptr_path = os.path.join(dataset_path, 'indptr.dat')
    indices_path = os.path.join(dataset_path, 'indices.dat')
    indptr = np.fromfile(indptr_path, dtype=conf['indptr_dtype']).reshape(tuple(conf['indptr_shape']))
    indices = np.memmap(indices_path, mode='r', shape=tuple(conf['indices_shape']), dtype=conf['indices_dtype'])
    indptr = torch.from_numpy(indptr)
    indices = torch.from_numpy(indices)

    conf_path = os.path.join(dataset_path, 'conf.json')
    conf = json.load(open(conf_path, 'r'))
    num_nodes = conf['num_nodes']
    # num_nodes = dataset.num_nodes

    neighbor_cache = NeighborCache(args.neigh_cache_size, score, indptr, indices_path, num_nodes)
    del(score)
    print('Done!')

    print('Saving neighbor cache...')
    cache_filename = str(dataset_path) + '/nc_size_' + str(args.neigh_cache_size)
    neighbor_cache.save(neighbor_cache.cache.numpy(), cache_filename)
    cache_tbl_filename = str(dataset_path) + '/nctbl_size_' + str(args.neigh_cache_size)
    neighbor_cache.save(neighbor_cache.address_table.numpy(), cache_tbl_filename)
    print('Done!')


# Save neighbor cache
print('Save neighbor cache...')
save_neighbor_cache()
print('Done!')
