import torch
import torch.nn as nn

class CellTypeClassifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=10):
        """
        Cell Type Classifier Head 
        :param input_dim: Input Feature Dimension (SVD/Geneformer dim) 
        :param num_classes: Number of cell types
        """
        super().__init__()
        
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




