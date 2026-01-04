import torch
import numpy as np
import os
import pickle
from transformers import BertForMaskedLM, BertTokenizer
from spatial_module import SpatialFeatureFusion 
from imputation_module import GeneImputationHead
from annotation_module import CellTypeClassifier

class Geneformer_InferenceEngine:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.max_input_size = 2048 
        
        self.imputation_model = None 

        print(f"[Geneformer] Initializing (Device: {device})...")

        try:
            self.spatial_module = SpatialFeatureFusion(embed_dim=256).to(self.device)
        except Exception as e:
            print(f"Spatial module initialization warning: {e}")

    def load_imputation_mlp(self, model_path, output_dim):
        try:
            self.imputation_model = GeneImputationHead(input_dim=256, output_dim=output_dim)
            
            state_dict = torch.load(model_path, map_location=self.device)
            self.imputation_model.load_state_dict(state_dict)
            self.imputation_model.to(self.device)
            self.imputation_model.eval()
        except Exception as e:
            print(f"[SToFM] MLP Failed to load: {e}")
            self.imputation_model = None
    def predict_single_gene(self, cell_features, gene_idx):
        if self.imputation_model is None:
            print("MLP not loaded")
            return None

        tensor_feat = torch.tensor(cell_features, dtype=torch.float32).to(self.device)
        
        model = self.imputation_model
        
        with torch.no_grad():
            hidden = model.decoder[0](tensor_feat) # Linear
            hidden = model.decoder[1](hidden)      # Norm
            hidden = model.decoder[2](hidden)      # GELU
            hidden = model.decoder[3](hidden)      # Linear
            hidden = model.decoder[4](hidden)      # Norm
            hidden = model.decoder[5](hidden)      # GELU

            last_linear = model.decoder[6] 

            w = last_linear.weight[gene_idx] 
            b = last_linear.bias[gene_idx]
            
            # Perform the linear transformation manually : y = xW^T + b
            # [N, 512] @ [512] -> [N]
            logits = torch.matmul(hidden, w) + b

            output = torch.sigmoid(logits)
            
        return output.cpu().numpy()

    def load_pretrained_model(self, model_dir):
        """
        Load the Geneformer model
        """
        try:
            
            self.model = BertForMaskedLM.from_pretrained(model_dir, output_hidden_states=True)
            self.model.to(self.device)
            self.model.eval()
            hidden_size = self.model.config.hidden_size
            self.spatial_module = SpatialFeatureFusion(embed_dim=hidden_size).to(self.device)

            spatial_weights_path = os.path.join(os.path.dirname(__file__), "spatial_weights.pth")
            if hasattr(self.spatial_module, "load_weights"):
                self.spatial_module.load_weights(spatial_weights_path)

            vocab_file = os.path.join(model_dir, "token_dictionary.pkl")
            if not os.path.exists(vocab_file):
                vocab_file = os.path.join(model_dir, "token_dictionary_gc104M.pkl")

            if os.path.exists(vocab_file):
                with open(vocab_file, "rb") as f:
                    self.vocab = pickle.load(f)
                print(f"Successfully loaded local word list containing {len(self.vocab)} genes")
            else:
                try:
                    self.tokenizer = BertTokenizer.from_pretrained(model_dir)
                    self.vocab = self.tokenizer.vocab
                    print("Successfully loaded HuggingFace Tokenizer")
                except:
                    print("Warning: No thesaurus file found.") 
                    self.vocab = {}

            self.is_loaded = True

        except Exception as e:
            print(f"[Geneformer] Failed to load: {e}")

    def get_gene_embedding(self, input_ids):
        if len(input_ids) == 0:
            return None
            
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor, output_hidden_states=True)
            emb = torch.mean(outputs.hidden_states[-1], dim=1)
        return emb

    def compute_batch_spatial_attention(self, batch_features, batch_coords, batch_size=512):
        n_cells = batch_features.shape[0]
        all_weights = []
        self.se2_model.eval() 
        if not hasattr(self, 'se2_model'):

             from se2_module import SE2Transformer
             self.se2_model = SE2Transformer(input_dim=256).to(self.device)

        for i in range(0, n_cells, batch_size):
            end = min(i + batch_size, n_cells)
            feats = torch.tensor(batch_features[i:end], dtype=torch.float32).to(self.device)
            coords = torch.tensor(batch_coords[i:end], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                _, attn_weights = self.se2_model(feats, coords)
                avg_attn = torch.mean(attn_weights, dim=1)
                weights = avg_attn[:, 0, :].cpu().numpy()
                all_weights.append(weights)
        
        return np.vstack(all_weights)

    def predict_perturbation_with_spatial(self, expression_values, gene_names, coords, perturbation_type="KO", target_gene_id=None):

        if not self.is_loaded or not self.vocab:
            return np.ones(len(expression_values)), "Error: Model not loaded."

        gene_expr_pairs = list(zip(gene_names, expression_values))
        sorted_genes = sorted([g for g in gene_expr_pairs if g[1] > 0], key=lambda x: x[1], reverse=True)
        top_genes = sorted_genes[:self.max_input_size]

        original_token_ids = []
        for g_id, _ in top_genes:
            if g_id == "Unknown": continue
            if g_id in self.vocab:
                original_token_ids.append(self.vocab[g_id])
        
        if len(original_token_ids) < 5:
            return np.ones(len(expression_values)), "Error: Too few valid genes expressed."

        perturbed_token_ids = original_token_ids.copy()
        target_token = self.vocab.get(target_gene_id)

        if target_token is None:
            return np.ones(len(expression_values)), f"Gene '{target_gene_id}' not found in model vocabulary."

        if perturbation_type == "KO":
            if target_token in perturbed_token_ids:
                perturbed_token_ids.remove(target_token)
                print(f"[Geneformer] True knockout: {target_gene_id}")
            else:
                return np.ones(len(expression_values)), f"Gene {target_gene_id} not expressed, KO ignored."

        elif perturbation_type == "OE":
            if target_token in perturbed_token_ids:
                perturbed_token_ids.remove(target_token)
            perturbed_token_ids.insert(0, target_token)
            perturbed_token_ids = perturbed_token_ids[:self.max_input_size]
            print(f"[Geneformer] True overexpression: {target_gene_id}")

        gene_emb_orig = self.get_gene_embedding(original_token_ids)
        gene_emb_pert = self.get_gene_embedding(perturbed_token_ids)
        
        if gene_emb_orig is None or gene_emb_pert is None:
             return np.ones(len(expression_values)), "Error: Embedding calculation failed."


        coords_tensor = torch.tensor([coords], dtype=torch.float32).to(self.device)
        coords_norm = coords_tensor / 20000.0 
        
        stofm_emb_orig = self.spatial_module(gene_emb_orig, coords_norm)
        stofm_emb_pert = self.spatial_module(gene_emb_pert, coords_norm)

        cosine_sim = torch.nn.functional.cosine_similarity(stofm_emb_orig, stofm_emb_pert, dim=-1)
        distance = 1.0 - cosine_sim.mean().item()

        impact_factor = 1.0 + (distance * 200.0)
        
        print(f"[SToFM Deduction] Spatial Fusion Distance: {distance:.6f} -> Influence Factor: {impact_factor:.4f}")
        
        return np.full(len(expression_values), impact_factor), "Success"
    # ============================================Cell type annotation=====================================================================
    def load_annotation_model(self, model_path, label_path, input_dim=256):
        try:
            with open(label_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            num_classes = len(self.label_encoder.classes_)
            
            self.annotation_model = CellTypeClassifier(input_dim=input_dim, num_classes=num_classes)
            self.annotation_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.annotation_model.to(self.device)
            self.annotation_model.eval()
        except Exception as e:
            print(f"[SToFM] Annotation model loading failed: {e}")

    def predict_cell_types(self, features):
        if self.annotation_model is None: 
            print("Annotation Model Not Loaded")
            return None, None
        batch_size = 4096
        n_cells = features.shape[0]
        all_preds = []
        
        tensor_feat_all = torch.tensor(features, dtype=torch.float32)
        
        with torch.no_grad():
            for i in range(0, n_cells, batch_size):
                batch_feat = tensor_feat_all[i : i+batch_size].to(self.device)
                outputs = self.annotation_model(batch_feat)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.append(predicted.cpu().numpy())
                
        pred_ids = np.concatenate(all_preds)

        class_names = self.label_encoder.classes_.tolist()
        print("[AI] 全场细胞类型预测完毕...")
        return pred_ids, class_names
# ================================================== Semantic segmentation of organizational regions ==========================================================
    def load_region_model(self, model_path, label_path):
        
        with open(label_path, "rb") as f:
            self.region_names = pickle.load(f)
        
        num_classes = len(self.region_names)
        self.region_model = CellTypeClassifier(input_dim=256, num_classes=num_classes)
        self.region_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.region_model.to(self.device).eval()

    def predict_regions(self, features):
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.region_model(features_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds.tolist()



