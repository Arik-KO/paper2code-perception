"""
Simplified DETR encoder layer will be added here. This code is motivated from the simplified implementation of DETR.
"""


import torch
import torch.nn as nn


class DETREncoderLayer(nn.Module):
    def __init__(self, d_model = 256, nhead = 8, dim_feedforward = 2048, dropout = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout = dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, src, pos):
        Q = src+pos
        K = src+pos
        V = src
        attn_output, attn_weights = self.self_attn(Q,K,V)
        first_residual = self.norm1(src+self.dropout(attn_output))
        first_linear = self.linear1(first_residual)
        activation = self.activation(first_linear)
        dropout = self.dropout(activation)
        second_linear = self.linear2(dropout)
        second_residual = self.norm2(first_residual + self.dropout(second_linear))
        return second_residual



if __name__ == "__main__":
    encoder_layer = DETREncoderLayer(d_model=256, nhead=8)
    src = torch.randn(950, 1, 256)
    pos = torch.randn(950, 1, 256)
    output = encoder_layer(src, pos)
    print(f"Input shape:  {src.shape}")
    print(f"Output shape: {output.shape}")
