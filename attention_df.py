import torch
import math
import torch.nn.functional as F

def attention(quary, key, value, mask=None, dropout=None):
    d_k= quary.size(-1)
    scores = torch.matmul(quary, key.transpose(-1,-1))\
             /math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

