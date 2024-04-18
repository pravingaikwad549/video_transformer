from v1 import model_v
import torch
from v1 import MyPositionalEncoding, block, final_layer, residualconnection, forwardBlock
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import functional as F
from data_loader import load_data
import json
from tqdm import tqdm
import sys
from torch.utils.data import TensorDataset, DataLoader

print("Imported Libraries")
# Load data
parent_folder_path = r"/home/pravin/Desktop/rough/fanply/UCF_EXP"
dataset, labels, dict_lables = load_data(parent_folder_path)
dataset = dataset[:, :, :-1]

file_path =  r"/home/pravin/Desktop/rough/fanply/video_gpt/dict_lables.json"

with open(file_path, 'w') as json_file:
    json.dump(dict_lables, json_file)

# sys.exit()
log_dir = r"/home/pravin/Desktop/rough/fanply/video_gpt/runs/logs"  # Directory to save the logs
writer = SummaryWriter(log_dir=log_dir)


print("Label: ", labels.shape)
print("Dataset: ", dataset.shape)
print("Dict Labels: ", len(dict_lables))

# Set model parameters
input_dim = dataset.shape[2] # 1628
hidden_dim = 256
output_dim = len(dict_lables) # 98
num_layers = 4 
num_classes = len(dict_lables) # 98
seq_length = dataset.shape[1] # 100


print("Input Dim: ", input_dim)
print("Output Dim: ", output_dim)
print("Hidden Dim: ", hidden_dim)
print("Num Layers: ", num_layers)
print("Num Classes: ", num_classes)
print("Seq Length: ", seq_length)

num_epochs = 1
batch_size = 2
learning_rate = 1e-5

dataset = torch.tensor(dataset, dtype = torch.float32).to("cuda")
labels = torch.tensor(labels, dtype = torch.long).to("cuda")

train_dataset = TensorDataset(dataset, labels)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)

model = model_v(MyPositionalEncoding, block, final_layer, residualconnection, 
                forwardBlock, dataset.shape[-1], seq_length, h = int(4), 
                num_classes = num_classes).to("cuda")

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

log_dir = "logs"
writer = SummaryWriter(log_dir=log_dir)
# sys.exit()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_input, batch_labels in train_loader:
        optimizer.zero_grad()
        output = model(batch_input)
        loss = criteria(output, batch_labels)
        print("Loss: ", loss.item())
        writer.add_scalar("Loss", loss.item())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print("epoch loss",epoch_loss)
writer.close()
print("Training complete!")

torch.save(model.state_dict(), "video_transformer1.pth")
print("Model saved!")