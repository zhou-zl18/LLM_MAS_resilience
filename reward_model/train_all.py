import json
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
import mlflow
import setproctitle
setproctitle.setproctitle('GCN')


mlflow.set_experiment("reward_model")
mlflow.start_run(run_name="all") #######################

# hyper-parameters
lr = 0.001
batch_size = 256
num_epochs = 200
best_model_path = "./output/reward_model_all.pth"
print(f"lr: {lr}, batch_size: {batch_size}, num_epochs: {num_epochs}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

mlflow.log_params({
    "lr": lr,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
})

filepath = "../data"
with open(os.path.join(filepath, "reward_model_training_data.json"), "r") as f:
    all_data = json.load(f)
print(len(all_data))

all_data = [x for x in all_data if len(x['results']) == 5]
print(len(all_data))

'''
{'graph': 'ER_10_0.1', 'adj_matrix': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'question_prompt': 'Given the chess game "a2a4 a7a5 d2d3 e7e5 f2f3 g7g5 g2g4 h7h5 a1a3 b7b6 e2e4 f7f6 e1d2 h8h6 a3a1 b8a6 g4h5 e8e7 f1g2 f8g7 d1e1 a8a7 g1e2 a6b8 b2b3 c8b7", give one valid destination square for the chess piece at "d2". Give a one line explanation of why your destination square is a valid move. State your final answer in a new line with a 2 letter response following the regex [a-h][1-8].', 'results': [0.0, 0.0, 0.0, 0.0, 0.0]}
'''

# load data
# filepath = "."
# with open(os.path.join(filepath, "train_data.json"), "r") as f:
#     train_data = json.load(f)
# with open(os.path.join(filepath, "test_data.json"), "r") as f:
#     test_data = json.load(f)

# test_graphs = sorted(list(set([x['graph'] for x in test_data])))


embedding_model = SentenceTransformer('../sentence-transformers/all-MiniLM-L6-v2/')
def get_sentence_embedding(sentence): # output: (384,)
    embeddings = embedding_model.encode(sentence)
    return embeddings

# Set random seeds for reproducibility
import random
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def calculate_resilience(accs: list): # accs: accuracy for error_rate = 0, 0.2, 0.4, 0.6, 0.8
    orig_acc = accs[0]
    if orig_acc == 0:
        return 0
    drops = []
    for acc in accs:
        if acc < orig_acc:
            drops.append(orig_acc - acc)
        else:
            drops.append(0)
    resilience = (orig_acc - np.mean(drops)) / orig_acc
    return resilience

class GNNRegressor(torch.nn.Module):
    def __init__(self, input_dim=384 + 2, hidden_channels=128):
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


random.shuffle(all_data)
train_data = all_data[:int(len(all_data) * 0.8)]
test_data = all_data[int(len(all_data) * 0.8):]


# Convert data to graphs
train_graphs = [create_graph(data) for data in tqdm(train_data)]
test_graphs = [create_graph(data) for data in tqdm(test_data)]

# Create data loaders
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

# Initialize model and optimizer

model = GNNRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Binary cross entropy loss for multi-label classification
criterion = torch.nn.BCELoss()

# Training loop
best_accuracy = 0.0


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred, batch.y.view(-1, 5))  # Reshape target to match prediction shape
        
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation on test set
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            test_loss += criterion(pred, batch.y.view(-1, 5)).item()  # Reshape target to match prediction shape
            
            # Calculate accuracy
            pred_binary = (pred > 0.5).float()
            correct += (pred_binary == batch.y.view(-1, 5)).sum().item()
            total += batch.y.numel()
            pred_list.extend(pred.cpu().numpy().tolist())

    avg_train_loss = total_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {avg_train_loss:.4f}')
    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Save model if test accuracy is the best so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_path) #########################
        print(f"Best model saved with accuracy: {best_accuracy:.4f}")

    mlflow.log_metrics({
        "train_loss": avg_train_loss,
        "test_loss": avg_test_loss,
        "test_accuracy": accuracy,
    }, step=epoch)


mlflow.end_run()
