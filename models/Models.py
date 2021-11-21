import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from models.utils import Normalization
import fastwer
import contextlib

# from nnAudio.Spectrogram import MelSpectrogram
import pandas as pd

class simpleLSTM(nn.Module):
    def __init__(self,
                 spec_layer,
                 input_dim,
                 hidden_dim,
                 num_lstms,
                 output_dim,
                 ):
        super().__init__()
        
        self.spec_layer = spec_layer
        self.embedding = nn.Linear(input_dim,hidden_dim)
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=num_lstms, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        print(1, spec.shape)
        spec = spec.transpose(1,2) # (B, T, F)
        print(2,spec.shape)
        x = self.embedding(spec)
        print(3,x.shape)
        x, _ = self.bilstm(x)
        print(4, x.shape)
        pred = self.classifier(x)
        print(pred.shape)
        output = {"prediction": pred,
                  "spectrogram": spec}
        return output
    
    
class simpleLinear(nn.Module):
    def __init__(self,
                 spec_layer,
                 input_dim,
                 hidden_dim1,
                 hidden_dim2,
                 hidden_dim3,                 
                 output_dim,
                 ):
        super().__init__()
        
        self.spec_layer = spec_layer
        self.linear1 = nn.Linear(input_dim,hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1,hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2,hidden_dim3)        
        self.classifier = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)
        x = torch.relu(self.linear1(spec))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))        
        pred = self.classifier(x)
        
        output = {"prediction": pred,
                  "spectrogram": spec}
        return output
    
    
class CNN_LSTM(nn.Module):
    def __init__(self,
                 spec_layer,
                 norm_mode,
                 input_dim,
                 hidden_dim=768,
                 output_dim=88):
        super().__init__()
        
        self.spec_layer = spec_layer
        self.norm_layer = Normalization(mode=norm_mode)
        
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, hidden_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(hidden_dim // 16, hidden_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(hidden_dim // 16, hidden_dim // 8, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((hidden_dim // 8) * (input_dim // 4), hidden_dim),
            nn.Dropout(0.5)
        )
        
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True, num_layers=1, bidirectional=True)
        
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)
        spec = self.norm_layer(spec)
        spec = spec.unsqueeze(1) # (B, 1, T, F)

        x = self.cnn(spec) # (B, hidden_dim//8, T, F//4)
        x = x.transpose(1,2).flatten(2)
        x = self.fc(x) # (B, T, hidden_dim//8*F//4)
        x, _ = self.bilstm(x)
        
        pred = self.classifier(x)
        
        output = {"prediction": pred,
                  "spectrogram": spec}
        return output        
        
class CNN_LINEAR(nn.Module):
    def __init__(self,
                 spec_layer,
                 norm_mode,
                 input_dim,
                 hidden_dim2,
                 hidden_dim3,
                 hidden_dim=768,
                 output_dim=88):
        super().__init__()
        
        self.spec_layer = spec_layer
        self.norm_layer = Normalization(mode=norm_mode)
        
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, hidden_dim // 16, (3, 3), padding=1),
            nn.Tanh(),
            # layer 1
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(hidden_dim // 16, hidden_dim // 16, (3, 3), padding=1),
            nn.Tanh(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(hidden_dim // 16, hidden_dim // 8, (3, 3), padding=1),
            nn.Tanh(),
        )

        self.fc = nn.Sequential(
            nn.Linear((hidden_dim // 8) * (input_dim // 4), hidden_dim2),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.Linear(hidden_dim3, hidden_dim),
            nn.Softmax()
        )
        
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)
        spec = self.norm_layer(spec)
        spec1 = spec.unsqueeze(1) # (B, 1, T, F)

        x = self.cnn(spec1) # (B, hidden_dim//8, T, F//4)
        x = x.transpose(1,2).flatten(2)
        x = self.fc(x) # (B, T, hidden_dim//8*F//4)
        
        pred = self.classifier(x)
        
        output = {"prediction": pred,
                  "spectrogram": spec}
        return output  


class VGGish(nn.Module):
    def __init__(self,
                 spec_layer,
                 norm_mode,
                 input_dim,
                 output_dim=128):
        super().__init__()
        
        self.spec_layer = spec_layer
        self.norm_layer = Normalization(mode=norm_mode)
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))

        self.embeddings = nn.Sequential(
            nn.Linear(512*24, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
            nn.ReLU(inplace=True))

    def forward(self, x):
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)
        spec = self.norm_layer(spec)
        spec = spec.unsqueeze(1) # (B, 1, T, F)

        x = self.features(spec) 
        x = x.view(x.size(0),-1)
        pred = self.embeddings(x) 
        
        output = {"prediction": pred,
                  "spectrogram": spec}
        return output   


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        # x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 

class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_layer = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        # x = self.norm_layer(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        # x = self.norm_layer(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)

class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.norm_layer = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = self.norm_layer(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class DeepSpeechModel(nn.Module):
    def __init__(self, spec_layer, norm_mode, n_cnn_layers, n_rnn_layers, 
                    rnn_dim, n_feats, input_dim, output_dim,stride=2, dropout=0.1):
        super(DeepSpeechModel, self).__init__()
        n_feats = n_feats//2
        self.spec_layer = spec_layer
        self.norm_layer = Normalization(mode=norm_mode)
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, output_dim)
        )
    def forward(self, inp):
        spec = self.spec_layer(inp) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)
        spec = self.norm_layer(spec)
        spec = spec.unsqueeze(1) # (B, 1, T, F)

        x = self.cnn(spec)
        x = self.rescnn_layers(x)
        sizes = x.size()
        # x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        # x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        pred = self.classifier(x)

        output = {"prediction": pred,
                  "spectrogram": spec}
        return output