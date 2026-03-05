# models.py
import torch
import torch.nn as nn
import torchvision.models as models
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

# ---------------- CNN ----------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
    def forward(self, x):
        return self.backbone(x)

# ---------------- CNN TrustNet ----------------
class CNNTrustNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.attention = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(512+128,1)
    def forward(self, x):
        feat = self.backbone.conv1(x)
        feat = self.backbone.layer1(self.backbone.maxpool(self.backbone.relu(feat)))
        feat = self.backbone.layer2(feat)
        feat = self.backbone.layer3(feat)
        feat = self.backbone.layer4(feat)
        gap = nn.AdaptiveAvgPool2d(1)(feat)
        gap_flat = torch.flatten(gap,1)
        att = self.attention(feat)
        att_flat = torch.flatten(att,1)
        combined = torch.cat([gap_flat, att_flat], dim=1)
        out = self.fc(combined)
        return out

# ---------------- Simple GNN ----------------
class SimpleGNN(nn.Module):
    def __init__(self, in_feats=3, hidden=32, out_feats=1):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, out_feats)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# ---------------- Hybrid CNN + GNN ----------------
class HybridCNNGNN(nn.Module):
    def __init__(self, use_gnn=True, trustnet=True):
        super().__init__()
        self.use_gnn = use_gnn
        self.trustnet = trustnet
        if trustnet:
            self.cnn = CNNTrustNet()
        else:
            self.cnn = CNN()
        if use_gnn:
            self.gnn = SimpleGNN()
    def forward(self, x, g=None):
        cnn_out = self.cnn(x)
        if self.use_gnn and g is not None:
            x_feat = g.x
            edge_index = g.edge_index
            gnn_out = self.gnn(x_feat, edge_index)
            gnn_out = gnn_out.mean(dim=0, keepdim=True)  # aggregate
            return cnn_out + gnn_out
        return cnn_out