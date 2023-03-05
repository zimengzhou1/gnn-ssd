import os.path as osp
from typing import Callable, Optional

import torch

from torch_geometric.data import InMemoryDataset, TemporalData, download_url, extract_gz, Data


class OverFlowDataset(InMemoryDataset):
    url = 'https://snap.stanford.edu/data/sx-stackoverflow.txt.gz'
    name = 'stackoverflow'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return ['sx-stackoverflow.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)
        extract_gz(self.raw_dir + "/sx-stackoverflow.txt.gz", self.raw_dir)
        

    def process(self):
        import pandas as pd

        df = pd.read_csv(osp.join(self.raw_dir, 'sx-stackoverflow.txt'), header=None, sep = ' ')

        src = torch.from_numpy(df.iloc[:, 0].values).to(torch.long)
        dst = torch.from_numpy(df.iloc[:, 1].values).to(torch.long)
        #dst += int(src.max()) + 1
        t = torch.from_numpy(df.iloc[:, 2].values).to(torch.long)
        # y = torch.from_numpy(df.iloc[:, 3].values).to(torch.long)
        # msg = torch.from_numpy(df.iloc[:, 4:].values).to(torch.float)
        msg = torch.rand(len(df.index), 128).to(torch.long)

        graph_data = Data(x=msg, edge_index=torch.stack([src, dst], dim=0), edge_attr=t)

        # data = TemporalData(src=src, dst=dst, t=t, msg=msg)

        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)

        torch.save(self.collate([graph_data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
