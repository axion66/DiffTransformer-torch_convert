import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffAttn(nn.Module):
    def __init__(self,embed_dim,layer_index=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.25
        self.q12 = nn.Linear(embed_dim, 2*embed_dim, bias=False)
        self.k12 = nn.Linear(embed_dim, 2*embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_q1 = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_init = self.set_lambda(torch.Tensor([layer_index])) if layer_index is not None else 0.8
        self.norm = nn.RMSNorm(embed_dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x):
        """
            :TODO:
                :positional encoding
                :masking
                :MHA, flash_attn   

            input shape: (batch,tgt_len,embed_dim)
        """
        
        x = self.diff_attention(x)
        x = self.norm(x)
        x *= (1 - self.lambda_init)
        return x
    
    def set_lambda(self,layer_index):
        return 0.8 - 0.6* torch.exp(-0.3 * (layer_index))

    def diff_attention(self,x):

        q1,q2 = torch.chunk(self.q12(x),chunks=2,dim=-1)
        k1,k2 = torch.chunk(self.k12(x),chunks=2,dim=-1)
        v = self.v(x)

        
        attn1 = torch.bmm(q1,k1.transpose(-1,-2)) * self.scale
        attn2 = torch.bmm(q2,k2.transpose(-1,-2)) * self.scale

        # from: https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py#L23
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        difference = F.softmax(attn1,dim=-1) - lambda_full * F.softmax(attn2,dim=-1)

        return torch.bmm(difference,v)