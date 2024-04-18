from v1 import model_v
import torch
from v1 import MyPositionalEncoding, block, final_layer, residualconnection, forwardBlock
from torch import nn
from torch.nn import functional as F
from data_loader import load_data
import json
from tqdm import tqdm
import sys
from torch.utils.data import TensorDataset, DataLoader
import json
import mediapipe as mp
import cv2
import numpy as np
from collections import deque
import time

ctime = 0
ptime = 0
fps = 0

input_dim = 1628
hidden_dim = 256
output_dim = 98
num_layers = 4 
num_classes = 4
seq_length =  100

model_inference = model_v(MyPositionalEncoding, block, final_layer, residualconnection, forwardBlock, input_dim, seq_length, h = int(4), num_classes = num_classes)
model_inference.load_state_dict(torch.load(r"/home/pravin/Desktop/rough/fanply/video_transformer1.pth"))
model_inference.eval()
print("Model loaded successfully!")

file_path = r"/home/pravin/Desktop/rough/fanply/video_gpt/dict_lables.json"

with open(file_path, 'r') as json_file:
    dict_labels = json.load(json_file)

print("Labels loaded successfully!")
print(dict_labels)
x = torch.randn(1, 100, 1628)

output = model_inference(x)
print(output.shape)
action_label = dict_labels[str(output.argmax(dim=1).item())]
print(action_label)