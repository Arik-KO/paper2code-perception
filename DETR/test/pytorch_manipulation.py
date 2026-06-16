"""
This script is to practice pytorch manipulation so that we can effectively address relevant detr operation,
especially to process image featues for the next setps like switching the dimension, concatenate, as well as 
flattening.
"""

import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
from utils.logger import setup_logger

manipulator_logger = setup_logger(name = 'tensor_manipulator')

# use of unsqueeze
a = torch.randn(3,4)
b = a.unsqueeze(0)
c = a.unsqueeze(1)
d = a.unsqueeze(-1)

random_tensor = torch.rand(256)
edited_tensor = random_tensor.unsqueeze(0)
edited_tensor = edited_tensor.unsqueeze(-1)
edited_tensor = edited_tensor.unsqueeze(-1)

# for squeeze see print with a1 tensor
a1 = torch.randn(1,3,1,4,1)

# permute to reorder the dimensions of a tensor
raw_tensor = torch.randn(2,3,4)
permutaed_tensor = raw_tensor.permute(2,0,1)

#flatten is used to merge dimension where flatten(start_dim, end_dim)
a2 = torch.randn(2,3,4,5)


#repeat to copy data along dimensions
a3 = torch.tensor([ [1,2,3]])

# concatenation to be done along a dim other than that dim all other dimensions must be of same legnth.
tensor_a = torch.tensor(  [  [5,2], [4,2], [3,2], [7,2], [8,2]  ] )
tensor_b = torch.tensor( [  [1,3,3], [2, 4, 4], [10, 1, 1], [7, 9, 9], [11, 12, 12]  ]  )


if __name__ == "__main__":
    #print(a.shape)
    #print(b.shape)
    #print(c.shape)
    #print(d.shape)
    #print(edited_tensor.shape)
    #print(a1.squeeze(0).shape)
    #print(a1.squeeze(1).shape)
    #print(a1.squeeze(2).shape)
    #print(a1.squeeze().shape)
    #print(raw_tensor.shape)
    #print(permutaed_tensor.shape)
    #print(a2.flatten(0,1).shape)
    #print(a2.flatten(2,3).shape)
    #print(a2.flatten(1,3).shape)
    #print(a2.flatten(2).shape)
    #print(a3.repeat(1,4))
    #print()
    #print(a3.repeat(2,3))
    print(torch.cat([tensor_a, tensor_b], dim = -1))
