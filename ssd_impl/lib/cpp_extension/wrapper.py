import os
from torch.utils.cpp_extension import load

dir_path = os.path.dirname(os.path.realpath(__file__))

gather = load(name='gather', sources=[os.path.join(dir_path, 'gather.cpp')], extra_cflags=['-fopenmp', '-O2'], extra_ldflags=['-lgomp','-lrt'])
free = load(name='free', sources=[os.path.join(dir_path, 'free.cpp')], extra_cflags=['-O2'])

