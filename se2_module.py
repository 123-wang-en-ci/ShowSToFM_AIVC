import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value 
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # SE(2) Core: Spatial position encoder
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads) 
        )

    def forward(self, x, coords):
        """
        x: [Batch, N, Dim] 
        coords: [Batch, N, 2]
        return: output, attn_weights
        """
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        content_score = (q @ k.transpose(-2, -1)) * self.scale

        diff = coords.unsqueeze(2) - coords.unsqueeze(1) # [B, N, N, 2]
        dist_sq = torch.sum(diff ** 2, dim=-1, keepdim=True)
        dist = torch.sqrt(dist_sq + 1e-6) 

        spatial_bias = self.distance_encoder(dist)

        spatial_bias = spatial_bias.permute(0, 3, 1, 2)

        total_score = content_score + spatial_bias

        attn_weights = F.softmax(total_score, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out, attn_weights

class SE2Transformer(nn.Module):
    def __init__(self, input_dim=256, num_layers=2, num_heads=4):

        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim) 
        
        self.layers = nn.ModuleList([
            SpatialAttention(input_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, gene_embeddings, spatial_coords):
        """
        gene_embeddings: [Batch, N, Dim]
        spatial_coords: [Batch, N, 2]
        """
        x = self.input_proj(gene_embeddings)
        
        last_attn_weights = None
        
        for layer in self.layers:

            out, attn = layer(x, spatial_coords)
            x = self.norm(x + out)
            last_attn_weights = attn
            
        return x, last_attn_weights