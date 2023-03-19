import os
import numpy as np
import torch
import pandas as pd

from torch_geometric.data import (
    InMemoryDataset,
    download_url,
    extract_zip,
)

from sklearn import preprocessing

# url = ('https://alicloud-dev.oss-cn-hangzhou.aliyuncs.com/UserBehavior.csv.zip')
# raw_dir = './'

# path = download_url(url, raw_dir)
# extract_zip(path, raw_dir)
# os.remove(path)
cols=['User_Id','Item_Id','Category_Id','Behavior_type','Timestamp']
df = pd.read_csv('./UserBehavior.csv', names=cols)

start = 1511539200
end = 1512316799
df = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]

df = df.drop_duplicates()

data = df
data = data.sort_values('Timestamp')

print(data)

# Message passing in bipartite graphs?

src = torch.tensor(data['User_Id'].values, dtype=torch.long)
dst = torch.tensor(data['Item_Id'].values, dtype=torch.long)
t = torch.tensor(data['Timestamp'].values)

print("starting label encoding...")
total_nodes = torch.unique(torch.cat((src,dst))).tolist()
le = preprocessing.LabelEncoder()
le.fit(total_nodes)
print("fitted")
print(type(le.transform(src.tolist())))
src = torch.from_numpy(le.transform(src.tolist())).to(torch.long)
dst = torch.from_numpy(le.transform(dst.tolist())).to(torch.long)
print(type(src))
print(src.shape)
print("preprocessing encoding")

from torch_geometric.data import Data
ddata = Data(edge_index=torch.stack([src, dst], dim=0), edge_attr=t)

torch.save(ddata, 'taobao.pt')