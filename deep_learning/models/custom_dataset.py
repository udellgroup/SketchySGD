import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float)
        
        # Convert string labels to integer labels
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(y)
        self.y = torch.tensor(encoded_labels).long()

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# class CustomDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X.values.astype(np.float64), dtype=torch.float)
        
#         # Convert string labels to integer labels
#         encoder = LabelEncoder()
#         encoded_labels = encoder.fit_transform(y.values)
#         self.y = torch.tensor(encoded_labels).long()

#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]