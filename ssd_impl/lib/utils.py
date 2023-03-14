from lib.cpp_extension.wrapper import *


def tensor_free(t):
    free.tensor_free(t)


def gather_mmap(features, idx):
    return gather.gather_mmap(features, idx, features.shape[1])

def gather_ginex(feature_file, idx, feature_dim, cache):
    return gather.gather_ginex(feature_file, idx, feature_dim, cache.cache, cache.address_table)
