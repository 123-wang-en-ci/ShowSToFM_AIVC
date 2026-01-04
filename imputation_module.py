import torch
import torch.nn as nn

class GeneImputationHead(nn.Module):
    def __init__(self, input_dim=256, output_dim=20000):
        """ 
        SToFM MLP Decoder 
        :param input_dim: Input feature dimension (corresponding to Geneformer/PCA dimension) 
        :param output_dim: Number of output genes (corresponding to n_vars of h5ad)
        """
        super().__init__()
        
        # Typical "Bottleneck-Expansion" structure
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [Batch, input_dim] 
        return: [Batch, output_dim] 
        """
        return self.decoder(x)