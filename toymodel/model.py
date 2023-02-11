import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as strc


from utils.my_utils import convert_to_batch, bn3Tob3n, fourier_encode

#Encoder
'''
Pointnet like convolutional encoder with Batch norm and relu '''


# Decoder

'''
Suppose to be a simple MLP
'''