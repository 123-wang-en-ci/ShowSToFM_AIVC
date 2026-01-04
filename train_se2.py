import torch
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
import numpy as np
from se2_module import SE2Transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

H5AD_FILE = "train.h5ad"
SAVE_PATH = "se2_weights.pth"
EMBED_DIM = 256 
BATCH_SIZE = 64 
NEIGHBORS_K = 50

def get_local_patches(coords, features, k=NEIGHBORS_K, n_patches=1000):
    n_cells = coords.shape[0]
    indices = np.random.choice(n_cells, n_patches, replace=False)
    
    batch_coords = []
    batch_feats = []
    
    from scipy.spatial import KDTree
    tree = KDTree(coords)
    
    for idx in indices:
        dists, n_idxs = tree.query(coords[idx], k=k)

        local_coords = coords[n_idxs] - coords[idx] 
        
        batch_coords.append(local_coords)
        batch_feats.append(features[n_idxs])
        
    return torch.tensor(np.array(batch_coords), dtype=torch.float32), \
           torch.tensor(np.array(batch_feats), dtype=torch.float32)

def train():
    adata = sc.read_h5ad(H5AD_FILE)

    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
    else:
        coords = adata.X[:, :2]
        
    if hasattr(adata.X, "toarray"): X = adata.X.toarray()
    else: X = adata.X
    
    pca = PCA(n_components=EMBED_DIM)
    features = pca.fit_transform(X)

    model = SE2Transformer(input_dim=EMBED_DIM, num_layers=2, num_heads=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(500):
        batch_coords, batch_feats = get_local_patches(coords, features)
        
        optimizer.zero_grad()

        recon_feats, _ = model(batch_feats, batch_coords)

        loss = criterion(recon_feats, batch_feats)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            
    torch.save(model.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    train()