import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import KDTree


class CellTypeClassifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=10):
        super(CellTypeClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),   
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes) 
        )

    def forward(self, x):
        return self.classifier(x)

def train_region_model():
    file_path = "Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad"
    adata = sc.read_h5ad(file_path)
    
    target_col = 'clust_annot'
    if target_col not in adata.obs.columns:
        print(f"Error: The label column cannot be found {target_col}")
        return
    adata = adata[~adata.obs[target_col].isna()].copy()
    
    X_log = np.log1p(adata.X)
    if hasattr(X_log, "toarray"): X_log = X_log.toarray()
    
    svd = TruncatedSVD(n_components=256, random_state=42)
    features = svd.fit_transform(X_log)

    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
    else:
        coords = adata.obsm['X_spatial']
    
    print(f"[Train] Feature extraction is complete: {features.shape}")

    le = LabelEncoder()
    labels = le.fit_transform(adata.obs[target_col])
    num_classes = len(le.classes_)
    print(f"[Train] Number of categories: {num_classes} -> {le.classes_}")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.15, random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellTypeClassifier(input_dim=256, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")


    torch.save(model.state_dict(), "region_segmentation_model.pth")
    with open("region_labels.pkl", "wb") as f:
        pickle.dump(le.classes_, f)

    with open("svd_model.pkl", "wb") as f:
        pickle.dump(svd, f)


if __name__ == "__main__":
    train_region_model()