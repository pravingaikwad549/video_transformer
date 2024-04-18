import os
import torch
from collections import defaultdict

def load_data(parent_folder_path):
    all_data = []
    all_labels = []
    folders = os.listdir(parent_folder_path)
    dict_labels = defaultdict()
    for folder_idx, folder in enumerate(folders):
        folder_path = os.path.join(parent_folder_path, folder)
        files = os.listdir(folder_path)
        
        for file in files:
            if file.endswith('.pt'):
                dict_labels[folder_idx] = folder
                tensor = torch.load(os.path.join(folder_path, file))
                num_datapoints = tensor.size(0)
                all_data.append(tensor)
                all_labels.extend([folder_idx] * num_datapoints)
    
    dataset = torch.cat(all_data, dim=0).to(torch.float32)
    labels = torch.tensor(all_labels, dtype=torch.long)
    
    return dataset, labels, dict_labels
