import torch
import matplotlib.pyplot as plt
import argparse
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser('Cache Graphing')
parser.add_argument('--path', type=str)

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

print("Using path: ", args.path[:-3])


def sum_intervals(vals, interval_size):
    num_intervals = len(vals) // interval_size
    interval_sums = []
    pbar = tqdm(total=num_intervals)
    for i in range(num_intervals):
        start_index = i * interval_size
        end_index = start_index + interval_size
        interval = vals[start_index:end_index]
        interval_sum = sum(interval) / interval_size
        interval_sums.append(interval_sum)
        pbar.update(1)
    pbar.close()
    return interval_sums

def getHitRate(stats):
    return sum(stats)/len(stats)

t = torch.load(args.path)
t = list(t)

interval_size = 20000
intervals_dynamic = sum_intervals(t, interval_size)
print(getHitRate(intervals_dynamic))


plt.plot(intervals_dynamic)
plt.title('Average cache hit per ' + str(interval_size) + ' requests')
plt.ylabel('Cache hit percentage')
plt.xlabel('Iterations')
plt.savefig(args.path[:-3] + '.png')