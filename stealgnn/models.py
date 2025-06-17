class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class CosineGenerator(nn.Module):
    def __init__(self, num_nodes, feature_dim, threshold):
        super(CosineGenerator, self).__init__()
        self.x_fake = nn.Parameter(torch.randn((num_nodes, feature_dim)))
        self.threshold = threshold

    def forward(self, threshold):
        x_norm = F.normalize(self.x_fake, dim=1)
        sim_matrix = torch.matmul(x_norm, x_norm.T)
        adj = (sim_matrix > threshold).float()
        adj.fill_diagonal_(0)
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        return self.x_fake, edge_index

class FullParameterizationGenerator(nn.Module):
    def __init__(self, num_nodes, feature_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim

        self.features = nn.Parameter(torch.randn(num_nodes, feature_dim))
        self.adj_logits = nn.Parameter(torch.randn(num_nodes, num_nodes))

    def forward(self, threshold):
        adj_sym = (self.adj_logits + self.adj_logits.T) / 2
        adj_thresh = (adj_sym > threshold).float()

        adj_thresh.fill_diagonal_(0)

        edge_index = adj_thresh.nonzero(as_tuple=False).t().contiguous()

        return self.features, edge_index
