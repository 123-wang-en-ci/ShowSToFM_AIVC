import torch
import torch.nn as nn
import scanpy as sc
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from annotation_module import CellTypeClassifier
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from scipy.sparse import issparse


TRAIN_FILE = "train.h5ad" 
TEST_FILE = "test.h5ad"  
MODEL_PATH = "annotation_mlp.pth"
LABEL_PATH = "label_encoder.pkl"
CELL_TYPE_COL = "cell_type"
INPUT_DIM = 256
BATCH_SIZE = 1024


def evaluate():
    print("[SToFM] (Cell Type Annotation)...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, TRAIN_FILE)
    test_path = os.path.join(base_dir, TEST_FILE)
    model_path = os.path.join(base_dir, MODEL_PATH)
    label_path = os.path.join(base_dir, LABEL_PATH)

    if not os.path.exists(test_path) or not os.path.exists(model_path):
        print(f"Error: File not found. Please check if {TEST_FILE} or {MODEL_PATH} exists.")
        return

    print(f"Utilizing the training set ({TRAIN_FILE}) to restore the feature space...")
    adata_train = sc.read_h5ad(train_path)
    
    if issparse(adata_train.X):
        X_train = adata_train.X.copy()
        X_train.data = np.log1p(X_train.data)
    else:
        X_train = np.log1p(adata_train.X)
        
    svd = TruncatedSVD(n_components=INPUT_DIM)
    svd.fit(X_train) 
    del adata_train, X_train 
    adata_test = sc.read_h5ad(test_path)
    
    if issparse(adata_test.X):
        X_test = adata_test.X.copy()
        X_test.data = np.log1p(X_test.data)
    else:
        X_test = np.log1p(adata_test.X)
    test_features = svd.transform(X_test)

    with open(label_path, "rb") as f:
        label_encoder = pickle.load(f)
    
    if CELL_TYPE_COL not in adata_test.obs:
        print(f"No columns found in the test set: {CELL_TYPE_COL}")
        return
        
    y_true_str = adata_test.obs[CELL_TYPE_COL].values
    y_true = label_encoder.transform(y_true_str)
    num_classes = len(label_encoder.classes_)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellTypeClassifier(input_dim=INPUT_DIM, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), BATCH_SIZE):
            batch = X_tensor[i:i+BATCH_SIZE]
            outputs = model(batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)

    accuracy = accuracy_score(y_true, y_pred)
    
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    f1_macro = f1_score(y_true, y_pred, average='macro')

    print("\n" * 2)
    print("=" * 60)
    print("Final Evaluation Results")
    print("=" * 60)
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"F1-Score (Weighted)    : {f1_weighted:.4f}  ")
    print(f"F1-Score (Macro)       : {f1_macro:.4f}")
    print("=" * 60)
    print("\n")

    print("Drawing a confusion matrix...")
    all_label_ids = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=all_label_ids)
    
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=False, cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Cell Type Annotation (Acc={accuracy:.4f}, F1={f1_weighted:.4f})")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("annotation_result.png", dpi=300)
    print("📈 图片已保存: annotation_result.png")

if __name__ == "__main__":
    evaluate()