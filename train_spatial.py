import torch
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
import numpy as np
import os
from spatial_module import SpatialFeatureFusion
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

H5AD_FILE = "train.h5ad"
SAVE_PATH = "spatial_weights.pth"
EPOCHS = 1000  
LR = 0.001 
EMBED_DIM = 256 

def train():
    if not os.path.exists(H5AD_FILE):
        print("h5ad file not found")
        return

    adata = sc.read_h5ad(H5AD_FILE)

    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
    else:
        coords = adata.X[:, :2]

    coord_scaler = MinMaxScaler()
    coords_norm = coord_scaler.fit_transform(coords)

    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = adata.X

    pca = PCA(n_components=EMBED_DIM)
    targets = pca.fit_transform(X)

    inputs = torch.tensor(coords_norm, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    model = SpatialFeatureFusion(input_dim=2, embed_dim=EMBED_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"Start training {EPOCHS} round...")
    model.train()
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model.get_spatial_embedding(inputs)

        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), SAVE_PATH)


if __name__ == "__main__":
    train()