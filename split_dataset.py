import scanpy as sc
import numpy as np
import os
from sklearn.model_selection import train_test_split

SOURCE_FILE = "Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad"
TRAIN_FILE = "train.h5ad"
TEST_FILE = "test.h5ad"
TEST_SIZE = 0.2  
RANDOM_STATE = 42 # Fixed random seeds ensure the same result each time


def split_data():
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(base_dir, SOURCE_FILE)
    train_path = os.path.join(base_dir, TRAIN_FILE)
    test_path = os.path.join(base_dir, TEST_FILE)

    if not os.path.exists(source_path):
        print(f"The source file cannot be found: {source_path}")
        return

    try:
        adata = sc.read_h5ad(source_path)
    except Exception as e:
        print(f"Read failed: {e}")
        return
        
    n_cells = adata.n_obs
    indices = np.arange(n_cells)

    train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    train_adata = adata[train_idx].copy()
    test_adata = adata[test_idx].copy()

    print(f"Train: {train_adata.n_obs} Cell")
    print(f"Test:  {test_adata.n_obs} Cell")

    train_adata.write(train_path)

    test_adata.write(test_path)

if __name__ == "__main__":
    split_data()