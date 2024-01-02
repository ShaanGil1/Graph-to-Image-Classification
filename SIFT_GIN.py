import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
import os
import pandas as pd
import numpy
import glob
import networkx as nx
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv
from tqdm import tqdm as tqdm_bar
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.nn import global_mean_pool

class CustomGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
        self.root_folder = root
        self.categories = os.listdir(self.root_folder)

    @property
    def raw_file_names(self):
        file_names = []
        for label in os.listdir(self.root):
            curr_path = os.path.join(self.root, label)
            edge_folder = os.path.join(curr_path, "edges")
            node_folder = os.path.join(curr_path, "nodes")
            for graph in os.listdir(edge_folder):
                edge_path = os.path.join(edge_folder, graph)
                node_name = graph[:-10]
                node_name += ".nodes.csv"
                node_path = os.path.join(node_folder, node_name)
                file_names.append((edge_path, node_path, label))
        return file_names

    def len(self):
        return len(self.raw_file_names)
    
    def class_mapping(self, label):
        mapping = {'n01440764': 0,
                   'n02102040': 1,
                   'n02979186': 2,
                   'n03000684': 3,
                   'n03028079': 4,
                   'n03394916': 5,
                   'n03417042': 6,
                   'n03425413': 7,
                   'n03445777': 8,
                   'n03888257': 9}
        return mapping[label]

    def get(self, idx):
        edge_path, node_path, label = self.raw_file_names[idx]

        edges = pd.read_csv(edge_path)
        nodes = pd.read_csv(node_path)
        nodes = nodes.drop(columns=["id"])

        x = torch.tensor(nodes.values, dtype=torch.float)
        edge_index = torch.tensor(edges[['node_1', 'node_2']].values, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edges['weight'].values, dtype=torch.float)

        label = self.class_mapping(label)
        label = torch.tensor(label)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=label)
        return data

base_path = "imagenette2_sift_knn"
train_root = base_path + "\\train"
val_root = base_path + "\\val"
train_dataset = CustomGraphDataset(train_root)
val_dataset = CustomGraphDataset(val_root)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        self.lin1 = Linear(out_channels, 32)
        self.lin2 = Linear(32, num_classes)
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = F.dropout(x, p=.5)
        x = self.lin2(x)

        return x

from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
        self.lin = nn.Linear(out_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.lin(x)
        return x
from torch_geometric.nn import ChebConv, global_mean_pool

class ChebGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, K=2):
        super(ChebGCN, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.conv3 = ChebConv(hidden_channels, out_channels, K)
        
        self.lin1 = nn.Linear(out_channels, 32)
        self.lin2 = nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x
    
class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(GraphSAGEModel, self).__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

        self.lin1 = Linear(out_channels, int(out_channels/2))
        self.lin2 = Linear(int(out_channels/2), num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = F.dropout(x, p=.5)
        x = self.lin2(x)
        return x
from torch_geometric.nn import GINConv, MLP

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(GIN, self).__init__()
        mlp1 = MLP([in_channels, hidden_channels, hidden_channels])
        mlp2 = MLP([hidden_channels, hidden_channels, out_channels])

        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.lin1 = nn.Linear(out_channels, int(out_channels/2))
        self.lin2 = nn.Linear(int(out_channels/2), num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.lin1(x)
        x = F.dropout(x, p=.5)
        x = self.lin2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

model = GIN(in_channels=train_dataset.num_features, hidden_channels=1024, out_channels=512, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)
    
# Training loop
num_epochs = 40
best_acc = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_samples = 0
    for batch in tqdm_bar(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        
        output = model(batch)

        loss = F.cross_entropy(output, batch.y)
        
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        pred = torch.argmax(F.sigmoid(output), dim=1)
        train_correct += (pred == batch.y).sum().item()
        train_samples += len(batch.y)

    train_loss = total_train_loss / len(train_loader)
    train_acc = train_correct/train_samples
    
    # Validation
    model.eval()
    val_correct = 0
    val_samples = 0
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm_bar(val_loader, desc='Validation', leave=False):
            batch = batch.to(device)

            output = model(batch)

            loss = F.cross_entropy(output, batch.y)
            total_val_loss += loss.item()

            pred = torch.argmax(F.sigmoid(output), dim=1)
            val_correct += (pred == batch.y).sum().item()
            val_samples += len(batch.y)

    val_loss = total_val_loss / len(val_loader)
    val_accuracy = val_correct / val_samples
    
    # Save the best model
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), "SIFT_GIN.pth")
    
    print(f"Epoch {epoch + 1}: Train Acc: {train_acc}, Train Loss: {train_loss}, Val Acc: {val_accuracy}, Val Loss: {val_loss}")
    