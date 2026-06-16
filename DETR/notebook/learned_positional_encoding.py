"""
positional encoding simple version has been implemented here. The positional encodings that added here will be learnable parameter,
rather than the sine cosine formula for encoding.

"""



import torch
import torch.nn as nn
import logging
import os
import sys



class PositionalEncoding2d(nn.Module):
    def __init__(self, hidden_dim = 256, max_position = 50):
        super().__init__()

        #learnable parameters for encodings
        
        self.row_embed = nn.Parameter(torch.rand(max_position, hidden_dim //2))
        self.col_embed = nn.Parameter(torch.rand(max_position, hidden_dim//2))
        
    def forward(self, H, W):
        sliced_col_embed = self.col_embed[:W]
        sliced_row_embed = self.row_embed[:H]
        expanded_col_embed = sliced_col_embed.unsqueeze(0).repeat(H,1,1)
        expanded_row_embed = sliced_row_embed.unsqueeze(1).repeat(1,W,1)
        embeddings = torch.cat([expanded_col_embed, expanded_row_embed], dim=-1)
        flattening = embeddings.flatten(0,1)
        return flattening.unsqueeze(1)

if __name__ == "__main__":
    pos_encoder = PositionalEncoding2d(256, 50)
    pos = pos_encoder(25, 38)
    print(f"Output shape: {pos.shape}")

