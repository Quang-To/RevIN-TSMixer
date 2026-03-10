import torch
import torch.nn as nn

class TimeMixingLayer(nn.Module):
    def __init__(self, seq_len: int, dropout: float):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(seq_len)  
        
        self.mlp_time = nn.Sequential(
            nn.Linear(seq_len, seq_len), 
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x                          
        x = self.batch_norm(x)            
        x = x.transpose(1, 2)               
        x = self.mlp_time(x)                
        x = x.transpose(1, 2)               
        x = x + residual                    
        return x