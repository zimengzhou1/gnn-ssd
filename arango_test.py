import torch
import pandas
from torch_geometric.datasets import FakeHeteroDataset

from arango import ArangoClient  # Python-Arango driver

from adbpyg_adapter import ADBPyG_Adapter, ADBPyG_Controller
from adbpyg_adapter.encoders import IdentityEncoder, CategoricalEncoder
from overflowDataset import OverFlowDataset

path = '/mnt/raid0nvme1/zz/data/' + 'overflow'
dataset = OverFlowDataset(path)
data = dataset[0]
n1 = torch.unique(data.edge_index[0])
n2 = torch.unique(data.edge_index[1])
total_nodes = torch.unique(torch.cat((n1,n2)))
data.x = total_nodes

from arango import ArangoClient
import base64

encodedCA = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSURHRENDQWdDZ0F3SUJBZ0lRYm52a1BPbHhXUEkydG9TTU1oWHFXVEFOQmdrcWhraUc5dzBCQVFzRkFEQW0KTVJFd0R3WURWUVFLRXdoQmNtRnVaMjlFUWpFUk1BOEdBMVVFQXhNSVFYSmhibWR2UkVJd0hoY05Nak13TXpJMwpNVE16TVRJeldoY05Namd3TXpJMU1UTXpNVEl6V2pBbU1SRXdEd1lEVlFRS0V3aEJjbUZ1WjI5RVFqRVJNQThHCkExVUVBeE1JUVhKaGJtZHZSRUl3Z2dFaU1BMEdDU3FHU0liM0RRRUJBUVVBQTRJQkR3QXdnZ0VLQW9JQkFRRDMKQ1BTRjM2bml5ZVNDTTFoNUIrVjdHc1A3QWl1SC9FR3M0bGE3TWx1RThJMEtJM3J6ZGlKTGVYaHZjRE9abHBITQpYWm5PeW81N3JZZWZRZG43WEpLekNIdTA4Wkg5c1NVWjY3Umx0QjZqVHZOYzg3a1BBdTY3TWtDSkpCT1dhbkVFCmZLVE9DdGsrYkVUbmdFVGR5dlVuYm90QTdtQU5wY1Z2VUhjUUNGVTFkRUs0S2dJbnBpZjdhMWJxTHVwNnVzZEoKSFF3VDNHS2JablN3L3F4S0FZU1kwNUE4MHJuTEVsOXVVQTAwbFJjazAvOXRMMmdJbWRxRW53VFZsei9IcWh1QwpaZjBJb09qVGpBaGd2MWxkbVdWdUNRblUycmVxWEozVlc4amp0NmdHQWhGZ0VmQW9tK0RYS29aRTA1NkphUW1VCmNla014L1ZtcXZDSWRBSTVlOS9MQWdNQkFBR2pRakJBTUE0R0ExVWREd0VCL3dRRUF3SUNwREFQQmdOVkhSTUIKQWY4RUJUQURBUUgvTUIwR0ExVWREZ1FXQkJSeTJ3Uk5RVFFTaTB3dy9KWnhYM1NvM1FyRk56QU5CZ2txaGtpRwo5dzBCQVFzRkFBT0NBUUVBUmx5VUo3L29hQ0dRa2NPY3JrL1RyclNBaGFsS0pBSEZiY09BcHZUcVpZc0xzYzdLClk5VEtvSW5WMmNqVkQwUnBLQ09DZ1hFcTQzMjdoR1ZlYSt1czh5WlBNTUJxQ0R5M3hEZXhrN016ZmJETVVoT1IKK1NwWFlHSnd3S1lQYTBoeWNNcmFrc2EreUl3ZzlLUjIrcndqQlNrNjlhY05VZ1R3dkZ6clJrQXd6R1kyamQ0egpDTEhMSnlhNHMwQ2tYTXlkR3NzUXZGQmV1djRKMjJyYUdLUVhSU20ybXJNamZFWjVzYmJWd2E4cjM1c0NpL3FDCmhCTUM5em13V29qL3VBWlQ2WDZJQklUV1IwSDE3c0dvSTZsNmlSZ2JEbDhuTnZLdFVZbWV5MkhCc0JXZGdHYkUKN3k0ekhtU2ltOGFXZm5yVFBoc0NwcFhCcDZreTZVa3lyd0RDMmc9PQotLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0tCg=="
try:
    file_content = base64.b64decode(encodedCA)
    with open("cert_file.crt", "w+") as f:
        f.write(file_content.decode("utf-8"))
except Exception as e:
    print(str(e))
    exit(1)

client = ArangoClient(
    hosts="https://0c561279bd83.arangodb.cloud:18529", verify_override="cert_file.crt"
)

sys_db = client.db("_system", username="root", password="nWqp1i1QmCRzUpUis29L")

# Note that ArangoGraph Insights Platform runs deployments in a cluster configuration.
# To achieve the best possible availability, your client application has to handle
# connection failures by retrying operations if needed.
print("ArangoDB:", sys_db.version())

adbpyg_adapter = ADBPyG_Adapter(sys_db)

adb_g = adbpyg_adapter.pyg_to_arangodb("overflow", data)