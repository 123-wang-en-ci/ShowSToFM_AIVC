import torch
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
import numpy as np
import os
from imputation_module import GeneImputationHead
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import issparse

H5AD_FILE = "train.h5ad"
SAVE_PATH = "imputation_mlp.pth"
INPUT_DIM = 256
EPOCHS = 300 
BATCH_SIZE = 64
LR = 1e-3
MASK_RATIO = 0.15 
MAX_TRAIN_CELLS = 20000 

def train():
   
    if not os.path.exists(H5AD_FILE):
        print(f"Not found {H5AD_FILE}")
        return

    try:
        adata = sc.read_h5ad(H5AD_FILE, backed='r')
    except Exception as e:
        print(f"data load failed: {e}") 
        print("Tip: Please try closing the running server.py to free up memory." )
        return

    n_cells = adata.n_obs
    n_genes = adata.n_vars
    if n_cells > MAX_TRAIN_CELLS:
        indices = np.random.choice(n_cells, MAX_TRAIN_CELLS, replace=False)
        indices.sort()

        X_matrix = adata[indices].X
    else:
        X_matrix = adata.to_memory().X

    if issparse(X_matrix): 
        pass 

    # Pre-computed SVD model (feature extraction)
    svd = TruncatedSVD(n_components=INPUT_DIM)
    if issparse(X_matrix):
        X_log = X_matrix.copy()
        X_log.data = np.log1p(X_log.data)
        svd.fit(X_log)
        features = svd.transform(X_log)
    else:
        X_log = np.log1p(X_matrix)
        svd.fit(X_log)
        features = svd.transform(X_log)

    model = GeneImputationHead(input_dim=INPUT_DIM, output_dim=n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    model.train()
    train_indices = np.arange(X_matrix.shape[0])
    
    for epoch in range(EPOCHS):
        batch_idx = np.random.choice(train_indices, BATCH_SIZE)

        raw_batch = X_matrix[batch_idx]
        if issparse(raw_batch): raw_batch = raw_batch.toarray()

        target_tensor = torch.tensor(np.log1p(raw_batch), dtype=torch.float32).to(device)

        mask = torch.bernoulli(torch.full_like(target_tensor, 1 - MASK_RATIO)).to(device)

        masked_input_np = np.log1p(raw_batch) * mask.cpu().numpy()

        batch_features = svd.transform(masked_input_np)
        feature_tensor = torch.tensor(batch_features, dtype=torch.float32).to(device)
        
        optimizer.zero_grad()

        outputs = model(feature_tensor)

        loss = criterion(outputs, target_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), SAVE_PATH)


if __name__ == "__main__":
    train()