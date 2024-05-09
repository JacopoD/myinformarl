import torch.nn as nn
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class GNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.information_aggregation = UniMPGNN(
            3 * self.dim + 3,  # (rel_pos, vel, rel_goal) * self.dim + entity embedding
            agg_out_channels
        )

        pass

    def forward(self, node_obs: torch.Tensor, adj: torch.Tensor):
        data = []
        for i in range(adj.shape[0]):
            edge_index, edge_attr = self.parse_adj(adj[i])
            data.append(
                Data(x=node_obs[i], edge_index=edge_index, edge_attr=edge_attr)
            )
        loader = DataLoader(data, shuffle=False, batch_size=adj.shape[0])
        data = next(iter(loader))
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        


    def parse_adj(adj: torch.Tensor, sensing_radius: float):
        assert adj.dim() == 2
        
        masks = ((adj < sensing_radius) * (adj > 0)).to(torch.float64)

        adj_masked = masks * adj

        #nonzero edge indexes
        nz_edge_indexes = adj_masked.nonzero(as_tuple=True)

        #nonzero edge attributes
        nz_edge_attrs = adj_masked[nz_edge_indexes]

        return torch.stack(nz_edge_indexes, dim=0), nz_edge_attrs
    

class UniMPLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1):
        super(UniMPLayer, self).__init__(aggr='add')  # Use 'add' for aggregating messages.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        
        # Define the weights for the linear transformations
        self.w1 = nn.Linear(in_channels, out_channels)
        self.w2 = nn.Linear(in_channels, out_channels)
        self.w3 = nn.Linear(in_channels, out_channels)
        self.w4 = nn.Linear(in_channels, out_channels)
        self.w5 = nn.Linear(1, out_channels)  # Assuming edge features are 1-dimensional

        self.sqrt_d = torch.sqrt(torch.tensor(out_channels, dtype=torch.float))

    def forward(self, x, edge_index, edge_attr):
        # Start propagating messages
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_index_i, edge_attr, size_i):
        # Transform node features and edge features
        x_i_transformed = self.w3(x_i)
        x_j_transformed = self.w4(x_j)
        edge_attr_transformed = self.w5(edge_attr)

        # Compute attention scores
        scores = (x_i_transformed * (x_j_transformed + edge_attr_transformed)).sum(dim=-1)
        scores = scores / self.sqrt_d
        alpha = softmax(scores, edge_index_i, num_nodes=size_i)
        
        # Transform node features for message passing
        return alpha.view(-1, 1) * self.w2(x_j)

    def update(self, aggr_out, x):
        # Node updates based on its features and aggregated messages
        return self.w1(x) + aggr_out


class UniMPGNN(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(UniMPGNN, self).__init__()
        self.layer = UniMPLayer(in_channels, out_channels, heads)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        return self.layer(x, edge_index, edge_attr)
