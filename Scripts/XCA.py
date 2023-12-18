import torch
import torch.nn as nn

"""
The crucial idea is the replace the multi-head attention from L*L to feature*features which is the calculation of co-variate matrix
Therefore the method is so-called cross-covariate attention 
"""
class XCA_4D(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mask_ratio= 0.2


    def forward(self, x):
        # B, N, C = x.shape
        residual = x
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        _, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1) # (B,H,C,L)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature # B, H, C, L -> B,H,L,C ->  B, H, C, C which is the covariate matrix (CxC)
        m_r = torch.ones_like(attn) * self.mask_ratio

        # DropKey
        attn = attn + torch.bernoulli(m_r) * -1e12
        attn = attn.softmax(dim=-1) 

        # attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C) # B, H, C, L -> B, L, C
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = x+residual
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

if __name__ == '__main__':
    x  = torch.randn(4,256,64 ,64)
    net = XCA_4D(dim=256)
    y = net(x)
    print(y.shape)


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
        sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mask_ratio= 0.2
        self.dropkey = True
    def forward(self, x):

        residual = x
        B,N,C = x.shape
        # print('before qkv ',x.shape)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)  # (B,L,3,HEADS,D/HEADS)
        # print('qkv',qkv.shape)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1) # (B,H,C,L)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature # B, H, C, L -> B,H,L,C ->  B, H, C, C which is the covariate matrix (CxC)
        
        
        # DropKey
        if self.dropkey:
            m_r = torch.ones_like(attn) * self.mask_ratio
            attn = attn + torch.bernoulli(m_r) * -1e12
            attn = attn.softmax(dim=-1) 
        else:
            attn = attn.softmax(dim=-1) 
            attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C) # B, H, C, L -> B, L, C
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # x = x.transpose(1, 2).view(B, C, H, W)
        x = x+residual

        # print('last',x.shape)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

# if __name__ == '__main__':
#     xx  = torch.randn(4,240,240 )
#     net = XCA(240,num_heads=2,attn_drop=0.3,proj_drop=0.2)
#     yy = net(xx)
#     print(yy.shape)
  