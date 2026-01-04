import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import scanpy as sc
import pandas as pd
import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.sparse import issparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD 
import os
import datetime
from model_engine import Geneformer_InferenceEngine 

# H5AD_FILENAME = "Allen2022Molecular_aging_MsBrainAgingSpatialDonor_10_0.h5ad" 
H5AD_FILENAME = "Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad" 
# H5AD_FILENAME = "data.h5ad" 
CSV_FILENAME = "unity_cell_data.csv"
CELL_TYPE_COLUMN = "cell_type" 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GeneRequest(BaseModel):
    gene_name: str
    use_imputation: bool = False 

class PerturbRequest(BaseModel):
    target_id: str
    perturb_type: str = "KO"
    target_gene: str = "ENSMUSG00000037010"

class DataManager:
    def __init__(self):
        self.adata = None
        self.spatial_tree = None
        self.coords = None
        self.indices_map = None
        self.scaler = MinMaxScaler()
        
        self.cached_total_counts = None
        self.cached_features = None 
        self.current_view_gene = "RESET"
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.ai_engine = Geneformer_InferenceEngine(device="cpu") 
        model_dir = os.path.join(self.base_dir, "geneformer_model") 
        self.ai_engine.load_pretrained_model(model_dir) 

    def load_and_sync_data(self):
        print(f"[Backend] Load data: {H5AD_FILENAME} ...")
        h5ad_path = os.path.join(self.base_dir, H5AD_FILENAME)

        if not os.path.exists(h5ad_path):
            print(f"Error: File not found {h5ad_path}")
            return

        self.adata = sc.read_h5ad(h5ad_path)
        print("Index example:", self.adata.var_names[:5]) 
        print("List of gene lists:", self.adata.var.columns)

        if self.adata is not None:
            top_10_genes = list(self.adata.var_names[:20])
            print(f"[Genetic Testing] The top 20 genes in the dataset:: {top_10_genes}")
            print(f"[Statistics] Total Cells: {self.adata.n_obs}, Total Genes: {self.adata.n_vars}")
        if not self.adata.var_names[0].startswith("ENSG"):
            found_id_col = None
            for col in self.adata.var.columns:
                if self.adata.var[col].astype(str).str.startswith("ENSG").all():
                    found_id_col = col
                    break
            if found_id_col:
                self.adata.var['ensembl_id'] = self.adata.var[found_id_col]
            else:
                self.adata.var['ensembl_id'] = "Unknown"
        else:
            self.adata.var['ensembl_id'] = self.adata.var_names

        if 'spatial' in self.adata.obsm:
            self.coords = self.adata.obsm['spatial']
        else:
            self.coords = self.adata.X[:, :2]

        if issparse(self.coords): self.coords = self.coords.toarray()
        if not isinstance(self.coords, np.ndarray): self.coords = np.array(self.coords)
            
        self.spatial_tree = KDTree(self.coords)
        self.indices_map = {idx: i for i, idx in enumerate(self.adata.obs.index)}

        if issparse(self.adata.X):
            raw_counts = np.ravel(self.adata.X.sum(axis=1))
        else:
            raw_counts = np.ravel(self.adata.X.sum(axis=1))
        self.cached_total_counts = self.scaler.fit_transform(raw_counts.reshape(-1, 1)).flatten()

        print("[Backend] Computed Features (SVD)...")
        try:
            if issparse(self.adata.X):
                X_for_pca = self.adata.X.copy()
                X_for_pca.data = np.log1p(X_for_pca.data)
            else:
                X_for_pca = np.log1p(self.adata.X)
            svd = TruncatedSVD(n_components=256)
            self.cached_features = svd.fit_transform(X_for_pca)
            if self.cached_features.shape[1] < 256:
                pad_width = 256 - self.cached_features.shape[1]
                self.cached_features = np.pad(self.cached_features, ((0,0), (0, pad_width)))

        except Exception as e:
            print(f"[Backend] SVD failed: {e}. Fallback to Zeros.")
            self.cached_features = np.zeros((self.adata.n_obs, 256), dtype=np.float32)

        mlp_path = os.path.join(self.base_dir, "imputation_mlp.pth")
        if os.path.exists(mlp_path):
            self.ai_engine.load_imputation_mlp(mlp_path, output_dim=self.adata.n_vars)
        else:
            print("Warning: 'imputation_mlp.pth' not found. AI Denoise will be unavailable.")
        anno_model_path = os.path.join(self.base_dir, "annotation_mlp.pth")
        anno_label_path = os.path.join(self.base_dir, "label_encoder.pkl")
        
        if os.path.exists(anno_model_path):
            self.ai_engine.load_annotation_model(anno_model_path, anno_label_path, input_dim=256)
            print("annotation_mlp.pth AI annotation model loads successfully.") 
        else:
            print("annotation_mlp.pth not found, AI annotation is not possible.") 

        try:
            dm.ai_engine.load_region_model(
                model_path="region_segmentation_model.pth", 
                label_path="region_labels.pkl"
            )
        except Exception as e:
            print(f"[Warning] Failed to load tissue area segmentation, please check the file path: {e}")
        print(f"[Backend] data loaded successfully.")

        self.export_csv_for_unity()

    def export_csv_for_unity(self):
        ids = self.adata.obs.index
    
        raw_x = self.coords[:, 0]
        raw_y = self.coords[:, 1]

        center_x = np.mean(raw_x)
        center_y = np.mean(raw_y)

        norm_x = raw_x - center_x
        norm_y = raw_y - center_y

        expression_norm = self.cached_total_counts 

        if CELL_TYPE_COLUMN in self.adata.obs:
            cell_type_names = self.adata.obs[CELL_TYPE_COLUMN].values
            cell_type_codes, uniques = pd.factorize(cell_type_names)
        else:
            cell_type_names = ["Unknown"] * len(ids)
            cell_type_codes = [0] * len(ids)

        df_export = pd.DataFrame({
            'id': ids, 
            'x': norm_x, 
            'y': norm_y, 
            'z': 0,
            'expression_level': expression_norm,
            'cell_type_id': cell_type_codes,
            'cell_type_name': cell_type_names
        })

        unity_csv_path = os.path.join(self.base_dir, "..", "Assets", "StreamingAssets", CSV_FILENAME)
        os.makedirs(os.path.dirname(unity_csv_path), exist_ok=True)

        try:
            df_export.to_csv(unity_csv_path, index=False)
            print(f"[Success] CSV Centralized saved to: {unity_csv_path}")
            print(f"[Info] Offset correction: X={center_x:.2f}, Y={center_y:.2f}")
        except Exception as e:
            print(f"[Failed] CSV save error: {e}")

    def save_imputed_data(self, gene_name):
        if gene_name == "RESET" or gene_name not in self.adata.var_names:
            return None, "Invalid gene name or RESET view cannot be saved."
  
        raw_values = self.get_gene_data(gene_name)

        imputed_values = self.impute_data(raw_values)

        df_result = pd.DataFrame({
            'cell_id': self.adata.obs.index,
            'x': self.coords[:, 0],
            'y': self.coords[:, 1],
            f'{gene_name}_imputed': imputed_values
        })

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"imputed_{gene_name}_{timestamp}.csv"
        save_path = os.path.join(self.base_dir, "..", "Assets", "StreamingAssets", filename)
        
        try:
            df_result.to_csv(save_path, index=False)
            return filename, "Success"
        except Exception as e:
            print(f"[save failed]: {e}")
            return None, str(e)

    def impute_data(self, gene_values):
        if self.ai_engine.imputation_model is None:
            print("MLP not loaded, skip imputation.") 
            return gene_values

        gene_name = self.current_view_gene
        if gene_name == "RESET": return gene_values
        
        try:
            gene_idx = self.adata.var_names.get_loc(gene_name)
        except:
            return gene_values

        features = self.cached_features 

        try:
            imputed_vals = self.ai_engine.predict_single_gene(features, gene_idx)
        except Exception as e:
            print(f"Prediction error: {e}")
            return gene_values
            
        final = np.where(gene_values > 0, gene_values, imputed_vals*10)
        return final

    def get_gene_data(self, gene_name):
        if gene_name.upper() in ["RESET", "TOTAL", "DEFAULT", "HARD_RESET"]:
            base_values = self.cached_total_counts 
        else:
            if gene_name not in self.adata.var_names: return None
            
            if self.adata.raw is not None:
                try: vals = self.adata.raw[:, gene_name].X
                except: vals = self.adata[:, gene_name].X
            else:
                vals = self.adata[:, gene_name].X
            
            if issparse(vals): vals = vals.toarray()
            base_values = self.scaler.fit_transform(vals.reshape(-1, 1)).flatten()

        return np.clip(base_values, 0.0, 5.0)

    def save_annotation_result(self):
        
        if self.cached_features is None:
            return None, "No features available. Please restart server."
        pred_ids, legend = self.ai_engine.predict_cell_types(self.cached_features)
        
        if pred_ids is None:
            return None, "Annotation model not loaded."
        predicted_names = np.array(legend)[pred_ids]

        data_dict = {
            'cell_id': self.adata.obs.index,
            'predicted_type_id': pred_ids,
            'predicted_type_name': predicted_names 
        }

        if CELL_TYPE_COLUMN in self.adata.obs:
            true_names = self.adata.obs[CELL_TYPE_COLUMN].values

            if len(true_names) == len(predicted_names):
                data_dict['ground_truth'] = true_names
                data_dict['is_correct'] = (predicted_names == true_names)
            else:
                print("Warning: Length mismatch between data and true values.") 

        df_result = pd.DataFrame(data_dict)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"annotation_result_{timestamp}.csv"
        save_path = os.path.join(self.base_dir, "..", "Assets", "StreamingAssets", filename)
        
        try:
            df_result.to_csv(save_path, index=False)
            print(f"[Saved successfully] File: {filename}")
            return filename, "Success"
        except Exception as e:
            print(f"[save failed]: {e}")
            return None, str(e)

    def save_region_result(self):
        
        if self.cached_features is None:
            return None, "No features available."

        region_ids = self.ai_engine.predict_regions(self.cached_features)

        region_names_map = np.array(dm.ai_engine.region_names)
        predicted_region_names = region_names_map[region_ids]

        # Prepare the saved data
        data_dict = {
            'cell_id': self.adata.obs.index,
            'x_coord': self.coords[:, 0],
            'y_coord': self.coords[:, 1],
            'region_id': region_ids,
            'region_name': predicted_region_names
        }

        df_result = pd.DataFrame(data_dict)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tissue_segmentation_{timestamp}.csv"
        save_path = os.path.join(self.base_dir, "..", "Assets", "StreamingAssets", filename)
        
        try:
            df_result.to_csv(save_path, index=False)
            print(f"[Saved successfully] File: {filename}")
            return filename, "Success"
        except Exception as e:
            print(f"[save failed]: {e}")
            return None, str(e)

dm = DataManager()


@app.on_event("startup")
def startup_event():
    dm.load_and_sync_data()

# ================= API =================
@app.post("/switch_gene")
async def switch_gene(req: GeneRequest):
    if dm.adata is None: raise HTTPException(500, "Data not loaded")
    
    target_gene = req.gene_name
    if target_gene in ["HARD_RESET", "RESET", "TOTAL"]:
        target_gene = "RESET"
    
    dm.current_view_gene = target_gene
    values = dm.get_gene_data(target_gene)
    
    if values is None: return {"status": "error", "message": "Gene not found"}
    
    # Default message
    msg = "View Switched"

    if req.use_imputation and target_gene != "RESET":
        values = dm.impute_data(values)
        msg = f"AI Imputation : {target_gene}"
    
    updates = []
    ids = dm.adata.obs.index
    for i, val in enumerate(values):
        updates.append({"id": str(ids[i]), "new_expr": round(float(val), 3)})
        
    # Returns JSON containing message
    return {"status": "success", "message": msg, "updates": updates}

#Save interpolation data interface
@app.post("/save_imputation")
async def save_imputation(req: GeneRequest):
    filename, msg = dm.save_imputed_data(req.gene_name)
    
    if filename:
        return {"status": "success", "message": f"Saved to {filename}"}
    else:
        return {"status": "error", "message": f"Save failed: {msg}"}

# Get AI annotation results
@app.post("/get_annotation")
async def get_annotation():
    if dm.adata is None: return {"status": "error", "message": "Data not loaded"}
    
    pred_ids, class_names = dm.ai_engine.predict_cell_types(dm.cached_features)
    
    if pred_ids is None:
        return {"status": "error", "message": "Model not loaded"}

    updates = []
    ids = dm.adata.obs.index
    for i, pid in enumerate(pred_ids):
        updates.append({
            "id": str(ids[i]),
            "pred_id": int(pid) 
        })
        
    return {
        "status": "success",
        "legend": class_names, 
        "updates": updates
    }

# Save comment results interface
@app.post("/save_annotation")
async def save_annotation():
    filename, msg = dm.save_annotation_result()
    if filename:
        return {"status": "success", "message": f"Saved to {filename}"}
    else:
        return {"status": "error", "message": f"Save failed: {msg}"}

# Get legend information for cell type annotations
@app.get("/annotation_legend")
async def get_annotation_legend():

    if dm.ai_engine.annotation_model is None:
        return {"status": "error", "message": "Annotation model not loaded"}

    if not hasattr(dm.ai_engine, 'label_encoder'):
        return {"status": "error", "message": "Label encoder not loaded"}
        
    class_names = dm.ai_engine.label_encoder.classes_.tolist()

    legend_data = []
    for idx, class_name in enumerate(class_names):
        legend_data.append({
            "id": idx,
            "name": class_name
        })
    
    return {
        "status": "success",
        "legend": legend_data
    }

@app.post("/get_tissue_regions")
async def get_tissue_regions():
    features = dm.cached_features 
    if features is None:
        return {"status": "error", "message": "Data not loaded"}

    preds = dm.ai_engine.predict_regions(features)

    if hasattr(preds, "tolist"):
        final_regions = preds.tolist()
    else:
        final_regions = preds

    if hasattr(dm.ai_engine.region_names, "tolist"):
        final_names = dm.ai_engine.region_names.tolist()
    else:
        final_names = list(dm.ai_engine.region_names)

    return {
        "status": "success",
        "regions": final_regions, 
        "names": final_names
    }
@app.post("/save_tissue_regions")
async def save_tissue_regions():
    filename, msg = dm.save_region_result()
    if filename:
        return {"status": "success", "message": f"Results saved to {filename}"}
    else:
        return {"status": "error", "message": f"Save failed: {msg}"}
# Placeholder interface to prevent front-end errors
@app.post("/perturb")
async def calculate_perturbation(req: PerturbRequest): return {} 
@app.post("/clear_perturbation")
async def clear_perturbation(): return {} 
@app.post("/save_manual")
async def save_manual(): return {} 
@app.post("/impute_all")
async def impute_all(): return {}
@app.post("/disable_imputation")
async def disable_imputation(): return {}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)