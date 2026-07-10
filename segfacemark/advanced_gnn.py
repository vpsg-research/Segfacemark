import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric import nn as gnn

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = Config()

def build_graph(images):
    """
    将图像 Batch 转换为 PyG 的 Batch 对象
    支持单通道 (C=1) 和多通道 (C=3)
    """
    batch_size, C, H, W = images.shape
    kernel_size = 3
    pad = kernel_size // 2
    
    # 预设偏移和权重
    offsets = torch.tensor([(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)], 
                          device=images.device, dtype=torch.long)
    distance_weights = torch.tensor([1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414], 
                                   device=images.device, dtype=torch.float32)
    inv_dist_weights = 1.0 / distance_weights
    
    # 填充并提取 Patch
    images_padded = F.pad(images, (pad, pad, pad, pad), mode='reflect')
    patches = images_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)  # [B, C, H, W, 3, 3]
    
    # 节点特征: [B, H*W, C*9]
    node_feats = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, H * W, C * 9)
    # 特征归一化
    node_feats = (node_feats - node_feats.mean(dim=2, keepdim=True)) / (node_feats.std(dim=2, keepdim=True) + 1e-6)
    
    graph_list = []
    # 网格坐标
    rows, cols = torch.meshgrid(torch.arange(H, device=images.device), 
                                torch.arange(W, device=images.device), indexing='ij')
    coords = torch.stack([rows.flatten(), cols.flatten()], dim=1)
    
    for i in range(batch_size):
        x = node_feats[i]
        
        neighbor_coords = coords.unsqueeze(1) + offsets.unsqueeze(0)
        nr, nc = neighbor_coords[..., 0], neighbor_coords[..., 1]
        
        valid = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        valid_indices = valid.nonzero()
        
        src = valid_indices[:, 0]
        offset_idx = valid_indices[:, 1]
        dst_coords = neighbor_coords[src, offset_idx]
        dst = dst_coords[:, 0] * W + dst_coords[:, 1]
        
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = inv_dist_weights[offset_idx]
        
        graph_list.append(Data(x=x, edge_index=edge_index, edge_weight=edge_weight))
    
    return Batch.from_data_list(graph_list).to(images.device)

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, out_dim=128):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = gnn.GCNConv(hidden_dim * 2, hidden_dim)
        self.global_pool = gnn.global_mean_pool
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight)
        
        global_feat = self.global_pool(x, batch) 
        return self.fc(global_feat)