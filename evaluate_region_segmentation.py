import scanpy as sc
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from annotation_module import CellTypeClassifier

def evaluate_region_segmentation(test_h5ad_path, model_path, label_path):
    if not os.path.exists(label_path):
        print(f"Error: Label file {label_path} not found")
        return
    with open(label_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    target_names = list(loaded_data.classes_) if hasattr(loaded_data, 'classes_') else list(loaded_data)
    name_to_id = {str(name): i for i, name in enumerate(target_names)}
    num_classes = len(target_names)
    print(f"The model supports {num_classes} categories.")

    print("A 256-dimensional feature space is being prepared..")
    adata_test = sc.read_h5ad(test_h5ad_path)
    X = adata_test.X
    if issparse(X): X = X.toarray()
    X_log = np.log1p(X)
    
    svd = TruncatedSVD(n_components=256, random_state=42)
    features = svd.fit_transform(X_log) 

    label_col = 'clust_annot' if 'clust_annot' in adata_test.obs else 'cell_type'

    valid_mask = adata_test.obs[label_col].astype(str).isin(target_names)
    if not valid_mask.any():
        print(f"Error: The labels in the test set do not match the labels saved by the model at all!")
        return
    
    y_true_str = adata_test.obs[label_col][valid_mask].astype(str).values
    y_true = np.array([name_to_id[name] for name in y_true_str])
    features_valid = features[valid_mask]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellTypeClassifier(input_dim=256, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    with torch.no_grad():
        inputs = torch.FloatTensor(features_valid).to(device)
        outputs = model(inputs)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted')
    
    print("\n" + "="*50)
    print(f"Final Result:")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score :   {f1_w:.4f}")
    print("="*50 + "\n")

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    sns.heatmap(cm_norm, annot=False, cmap='Blues')
    plt.title(f"Evaluation Acc: {acc:.4f}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("evaluation_report.png")

if __name__ == "__main__":
    evaluate_region_segmentation(
        "test.h5ad", 
        "region_segmentation_model.pth", 
        "region_labels.pkl"
    )