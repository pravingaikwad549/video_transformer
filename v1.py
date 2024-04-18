# d_model in our case is 1629 and seq length is 100
import math
import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F

class MyPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.seq_length = seq_length

        # creating a matrix of shape (seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)
        # creating a vector of shape (seq_length, 1)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1) # (seq_length, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_length, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)

class MultiheadAttnBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0

        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        # (batch_size, h, seq_length, d_k) x (batch_size, h, d_k, seq_length) = (batch_size, h, seq_length, seq_length)
        attention_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9) # (batch_size, h, seq_length, seq_length)
        attention_score = F.softmax(attention_score, dim=-1) # (batch_size, h, seq_length, seq_length)

        if dropout is not None: 
            attention_score = dropout(attention_score)
        return torch.matmul(attention_score, value), attention_score
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        query = self.w_q(q) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        key = self.w_k(k) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        value = self.w_v(v) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)

        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, h, d_k) --> (batch_size, h, seq_length, d_k) 
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, h, d_k) --> (batch_size, h, seq_length, d_k)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, h, d_k) --> (batch_size, h, seq_length, d_k)
        
        x, attention_score = MultiheadAttnBlock.attention(query, key, value, mask, self.dropout) # (batch_size, h, seq_length, d_k)
        x = x.permute(0,2,1,3).contiguous().view(x.size(0), -1, self.h * self.d_k) # (batch_size, h, seq_length, d_k) --> (batch_size, seq_length, h, d_k) --> (batch_size, seq_length, d_model)
        x = self.w_o(x) 
        return x # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)

class residualconnection(nn.Module):
    def __init__(self, d_model, h, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()
        self.attnblock = MultiheadAttnBlock(d_model, h)
    def forward(self, x):
        return self.norm(x) + self.dropout(self.attnblock(self.norm(x), self.norm(x), self.norm(x)))

class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))   
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class forwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class block(nn.Module):
    def __init__(self, res, fb, d_model, h):
        super().__init__()
        self.res = res(d_model, h)
        self.fb = fb(d_model)
    def forward(self, x):
        x = self.res(x)
        x = self.fb(x)
        return x
    
    
class final_layer(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1, hidden_dim =256):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, x):
        x = torch.sum(x, dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

    
class model_v(nn.Module):
    def __init__(self, pos, bl, fl, res, fb, d_model, seq_length, h, num_classes):
        super().__init__()
        self.pos = pos(d_model = d_model, seq_length = seq_length)
        self.bl =  bl(res, fb, d_model = d_model, h = h)
        self.fl =  fl(d_model = d_model, num_classes = num_classes)
    def forward(self, x):      
        x = self.pos(x)
        x = self.bl(self.bl(x))
        x = self.fl(x)
        return x


