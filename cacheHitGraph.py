import torch
import matplotlib.pyplot as plt
import os
import argparse
import sys
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser('Cache Graphing')
parser.add_argument('--path', type=str)
parser.add_argument('--dataset', type=str)

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

print("Using path: ", args.path[:-3])

def getHitRate(stats):
    return sum(stats)/len(stats)

def sum_intv_new(vals, interval_size):
  return np.nanmean(np.pad(vals.astype(float), (0, interval_size - vals.size%interval_size), mode='constant', constant_values=np.NaN).reshape(-1, interval_size), axis=1)

__file__ = os.path.abspath('')
print(__file__)
pathStart = __file__ + "/cache_data/" + args.dataset + "/"
t1 = torch.load(pathStart +  "LRU_" + args.dataset + args.path).numpy()
t2 = torch.load(pathStart +  "static_" + args.dataset + args.path).numpy()

interval_size = 20000
LRU_intervals = sum_intv_new(t1, interval_size)
static_intervals = sum_intv_new(t2, interval_size)
print("static hit rate: ", getHitRate(static_intervals))
print("LRU hit rate: ", getHitRate(LRU_intervals))

plt.plot(static_intervals, label='static')
plt.plot(LRU_intervals, label='LRU')
plt.title('Average cache hit per ' + str(interval_size) + ' requests')
plt.ylabel('Cache hit percentage')
plt.xlabel('Iterations')
plt.legend()
plt.savefig(pathStart + args.path[:-3] + '.png')