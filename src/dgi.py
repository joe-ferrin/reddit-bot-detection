import torch
from torch_geometric.nn import GCNConv, DeepGraphInfomax
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

# GCN encoder
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return F.relu(x)

# now a DGI class method
# # Corruption function
# def corruption(self, x, edge_index):
#     rng = torch.Generator()
#     rng.manual_seed(self.seed)
#     permutation = torch.randperm(x.size(0), generator=rng)
#     return x[permutation], edge_index

# class to create dgi model and return embeddings
class DGI:
    def __init__(self, data, hidden=128, learning_rate=0.001, epochs=100, seed=0):
        self.seed = seed # store seed
        self.data = data
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.encoder = GCNEncoder(data.num_features, out_channels=hidden)
        self.model = self.create_model()
    # Corruption function
    def corruption(self, x, edge_index):
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        permutation = torch.randperm(x.size(0), generator=rng)
        return x[permutation], edge_index
    
    # Create DGI model
    def create_model(self):
        # not needed, passed a class attribute
        # encoder = GCNEncoder(self.data.num_features, out_channels=self.hidden)

        model = DeepGraphInfomax(
            hidden_channels=self.hidden,
            encoder=self.encoder,
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=self.corruption
        )
        return model

    # Return embedding
    def embed(self):
        torch.manual_seed(self.seed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pos_z, neg_z, summary = self.model(self.data.x, self.data.edge_index)
            loss = self.model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self.model.eval()
        embeddings = self.model.encoder(self.data.x, self.data.edge_index).detach()


        # Convert to dataframe
        embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
        embeddings_df = pd.DataFrame(embeddings_np, columns=[f"emb_{i}" for i in range(embeddings_np.shape[1])])
        embeddings_df['author'] = self.data.node_order

        return embeddings_df

