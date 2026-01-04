import torch
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
import numpy as np
import pandas as pd
import os
import pickle
from annotation_module import CellTypeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse

H5AD_FILE = "train.h5ad"
SAVE_MODEL_PATH = "annotation_mlp.pth"
SAVE_LABEL_PATH = "label_encoder.pkl"
CELL_TYPE_COL = "cell_type"
INPUT_DIM = 256
EPOCHS = 300
BATCH_SIZE = 64
LR = 0.005

def train():

    if not os.path.exists(H5AD_FILE):
        print(f"{H5AD_FILE} not found, please run split_dataset.py first")
        return

    adata = sc.read_h5ad(H5AD_FILE)

    if CELL_TYPE_COL not in adata.obs:
        print(f" column name {CELL_TYPE_COL} does not exist! Available columns: {adata.obs.columns}")
        return

    raw_labels = adata.obs[CELL_TYPE_COL].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(raw_labels)
    num_classes = len(label_encoder.classes_)

    with open(SAVE_LABEL_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    if issparse(adata.X):
        X_matrix = adata.X.copy()
        np.log1p(X_matrix.data, out=X_matrix.data)
    else:
        X_matrix = np.log1p(adata.X)
        
    svd = TruncatedSVD(n_components=INPUT_DIM)
    features = svd.fit_transform(X_matrix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(device)

    model = CellTypeClassifier(input_dim=INPUT_DIM, num_classes=num_classes)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    indices = np.arange(adata.n_obs)
    
    for epoch in range(EPOCHS):
        batch_idx = np.random.choice(indices, BATCH_SIZE)
        
        batch_x = X_tensor[batch_idx]
        batch_y = y_tensor[batch_idx]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == batch_y).sum().item() / BATCH_SIZE
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Batch Acc: {acc:.2%}")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()