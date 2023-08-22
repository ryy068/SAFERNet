import math
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from typing import Sequence

# use DWC
class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
                
        self.shortcut = nn.Parameter(torch.eye(kernel_size).reshape(1, 1, kernel_size, kernel_size))
        self.shortcut.requires_grad = False
        
    def forward(self, x):
        return F.conv2d(x, self.conv.weight+self.shortcut, self.conv.bias, stride=1, padding=self.kernel_size//2, groups=self.dim) # equal to x + conv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True, downsample=False, kernel_size=5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        self.conv = ResDWC(hidden_features, 3)
        
    def forward(self, x,hw_shape): 
        B, _, C = x.shape
        H0, W0 = hw_shape
        x = torch.transpose(x, 1, 2) 
        x = x.view(B, C, H0, W0)	
           
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)

        x = x.view(B, C, H0*W0)
        x = torch.transpose(x, 1, 2) 
        return x


# Dual Attn FFN
class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.25, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)

        self.channel_mul_conv = nn.Sequential(
            nn.Linear(in_channels, reduce_channels),
            nn.LayerNorm(reduce_channels),
            nn.ReLU(inplace=True),  
            nn.Linear(reduce_channels, in_channels))

        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
    """
    def spatial_pool(self, x):
        batch, channel,L = x.size()
        H=int(math.sqrt(L))
        input_x = x
        x=x.view(batch, channel,H,H)
        # [N, C, H * W]
        # input_x = input_x.view(batch, channel, L)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, L)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)
      
        return context
    """
    def forward(self, x):
        ori_x = x 
        x = self.norm(x)
       
        x_global = torch.sigmoid(self.channel_mul_conv(x))
        x_global = x_global * x

        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  
        s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        s_attn = self.gate_fn(s_attn) 

        attn = c_attn * s_attn  
        return ori_x * attn


class BiAttnMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.attn = BiAttn(out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.attn(x)
        x = self.drop(x)
        return x