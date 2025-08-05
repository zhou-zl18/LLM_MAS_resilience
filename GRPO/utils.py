import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sentence_transformers import SentenceTransformer
import re
embedding_model = SentenceTransformer('./sentence-transformers/all-MiniLM-L6-v2/')
def get_sentence_embedding(sentence): # output: (384,)
    embeddings = embedding_model.encode(sentence)
    return embeddings

def calculate_resilience(accs: list): # accs: accuracy for error_rate = 0, 0.2, 0.4, 0.6, 0.8
    orig_acc = accs[0]
    if orig_acc == 0:
        return 0
    tmp = []
    for acc in accs:
        if acc < orig_acc:
            tmp.append(acc)
        else:
            tmp.append(orig_acc)
    resilience = (tmp[0] + 2 * np.sum(tmp[1:]) + 0)/ 10 / orig_acc
    return resilience

def extract_constraints(text):
    pattern = r"- N\s*=\s*(\d+)\s*agents.*?- M\s*=\s*(\d+)\s*"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        N = int(match.group(1))
        M = int(match.group(2))
        return N, M
    else:
        return None, None

class GNNRegressor(torch.nn.Module):
    def __init__(self, input_dim=384, hidden_channels=128):
        super(GNNRegressor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels//2)
        self.lin2 = torch.nn.Linear(hidden_channels//2, 5)
        self.dropout = torch.nn.Dropout(0.0)

    def forward(self, x, edge_index, batch):
        # x shape: (num_nodes, input_dim)
        # edge_index shape: (2, num_edges) 
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)  # (batch_size, hidden_channels)
        
        # MLP head
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)  # Binary classification for each dimension
        
        return x  # Shape: (batch_size, 5)
    
# Create graph dataset
def create_graph(data_dict):
    adj_matrix = np.array(data_dict['adj_matrix'])
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
    num_nodes = adj_matrix.shape[0]
    
    # Get node features from question prompt
    task_features = get_sentence_embedding(data_dict['question_prompt'])
    task_features = torch.tensor(task_features, dtype=torch.float).unsqueeze(0)
    task_features = task_features.repeat(num_nodes, 1)  # Repeat for each node

    # Set node features to [in degree, out degree] for each node
    in_degrees = np.sum(adj_matrix, axis=0)  # shape: (num_nodes,)
    out_degrees = np.sum(adj_matrix, axis=1)  # shape: (num_nodes,)
    node_features = np.stack([in_degrees, out_degrees], axis=1)  # shape: (num_nodes, 2)
    node_features = torch.tensor(node_features, dtype=torch.float)

    node_features = torch.cat([task_features, node_features], dim=1)
    
    # Create graph data object
    data = Data(
        x=node_features, 
        edge_index=edge_index,
        y=torch.tensor(data_dict['results'], dtype=torch.float)
    )
    return data