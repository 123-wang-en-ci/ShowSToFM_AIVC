import torch
import torch.nn as nn
import os

class SpatialFeatureFusion(nn.Module):
    def __init__(self, input_dim=2, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.spatial_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embed_dim) 
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, gene_embeddings, spatial_coords):
        spatial_emb = self.spatial_mlp(spatial_coords)
        if spatial_emb.dim() == 2:
            spatial_emb = spatial_emb.unsqueeze(1)
            
        return gene_embeddings + (self.fusion_weight * spatial_emb)
    
    def get_spatial_embedding(self, spatial_coords):
        """ 
        Export only the spatial part of the features (only for training scripts) 
        """
        return self.spatial_mlp(spatial_coords)

    def load_weights(self, weight_path):
        if os.path.exists(weight_path):
            try:
                device = next(self.parameters()).device
                state_dict = torch.load(weight_path, map_location=device)
                self.load_state_dict(state_dict)
                self.eval() 

                return True
            except Exception as e:
                print(f"[SpatialModule] Weight loading failed: {e}")
                return False
        else:
            print(f"[SpatialModule] No weight file found: {weight_path}, using random initialization.")
            return False