import torch
import numpy as np
import scanpy as sc
import os
import matplotlib.pyplot as plt
import seaborn as sns
from imputation_module import GeneImputationHead
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from tqdm import tqdm 


H5AD_FILE = "test.h5ad"
MODEL_PATH = "imputation_mlp.pth"
INPUT_DIM = 256
N_TEST_CELLS = -1     # -1 represents evaluating all data
MASK_RATIO = 0.15     

def evaluate():
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    h5ad_path = os.path.join(base_dir, H5AD_FILE)
    model_path = os.path.join(base_dir, MODEL_PATH)

    if not os.path.exists(h5ad_path) or not os.path.exists(model_path):
        print(f"{H5AD_FILE} or model weight file cannot be found. Please run it first split_dataset.py 和 train_imputation.py。")
        return

    adata = sc.read_h5ad(h5ad_path, backed='r')
    n_genes = adata.n_vars
    n_available = adata.n_obs

    if N_TEST_CELLS == -1 or N_TEST_CELLS >= n_available:
        print(f"All {n_available} cells in the test set are being evaluated...")
        indices = np.arange(n_available) 
    else:
        print(f"Randomly select {N_TEST_CELLS} cells from the test set for quick evaluation...")
        indices = np.random.choice(n_available, N_TEST_CELLS, replace=False)
        indices.sort() 

    X_test = adata[indices].X
    if issparse(X_test): 
        X_test = X_test.toarray()
    else:
        X_test = np.array(X_test)

    # Prepare the feature extractor (SVD)
    svd = TruncatedSVD(n_components=INPUT_DIM)
    X_log = np.log1p(X_test)
    features = svd.fit_transform(X_log)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeneImputationHead(input_dim=INPUT_DIM, output_dim=n_genes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tensor_X = torch.tensor(X_log, dtype=torch.float32).to(device)
    mask = torch.bernoulli(torch.full_like(tensor_X, 1 - MASK_RATIO)).to(device)

    masked_input = X_log * mask.cpu().numpy()

    masked_features = svd.transform(masked_input)
    masked_features_tensor = torch.tensor(masked_features, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(masked_features_tensor)
        
    predicted = outputs.cpu().numpy()
    
    print("Calculate statistical indicators...")
    mask_np = mask.cpu().numpy()
    masked_indices = np.where(mask_np == 0)

    X_log_norm = X_log / (X_log.max() + 1e-9)
    
    true_values = X_log_norm[masked_indices]
    pred_values = predicted[masked_indices]

    non_zero_mask = true_values > 0.01
    y_true = true_values[non_zero_mask]
    y_pred = pred_values[non_zero_mask]
    
    if len(y_true) == 0:
        print("Insufficient valid test samples")
        return
    pcc, _ = pearsonr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    print("\n" + "="*50)
    print("Evaluation on Full Test Set")
    print("="*50)
    print(f"Sample size: {len(indices)} cells")
    print(f"Number of Gene Points Tested: {len(y_true)}")
    print("-" * 40)
    print(f"Pearson Correlation (PCC):  {pcc:.4f}  ")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):     {mae:.4f} ")
    print("-" * 40)

    plt.figure(figsize=(8, 8))
    plot_sample_size = 5000
    if len(y_true) > plot_sample_size:
        sample_idx = np.random.choice(len(y_true), plot_sample_size, replace=False)
        plt_true = y_true[sample_idx]
        plt_pred = y_pred[sample_idx]
    else:
        plt_true = y_true
        plt_pred = y_pred
        
    sns.regplot(x=plt_true, y=plt_pred, scatter_kws={'s': 5, 'alpha': 0.3}, line_kws={'color': 'red'})
    plt.xlabel("True Expression (Normalized)")
    plt.ylabel("Predicted Expression (Imputed)")
    plt.title(f"Imputation Accuracy\nPCC={pcc:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")
    
    save_path = os.path.join(base_dir, "imputation_evaluation.png")
    plt.savefig(save_path)

if __name__ == "__main__":
    evaluate()