# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Some functions are based on the timm and 4M code bases
# https://github.com/huggingface/pytorch-image-models
# https://github.com/apple/ml-4m
# --------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm(nn.Module):
    """Custom implementation of LayerNorm with the option to disable the bias term."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_buffer("bias", torch.zeros(normalized_shape))

        # Normalized shape must be a tuple for F.layer_norm
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps=self.eps)


class Mlp(nn.Module):
    """
    MLP module with GELU activation.

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features (optional)
        out_features: Number of output features (optional)
        bias: Whether to include bias in the linear layers
    """
    def __init__(self, 
            in_features: int, 
            hidden_features: Optional[int] = None, 
            out_features: Optional[int] = None, 
            bias: bool = False,
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        ## Begining of My code 
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## Begining of My code 
        x=self.act(self.fc1(x))
        x=self.fc2(x)
        return x 


class Attention(nn.Module):
    """
    Multi-head self-attention module.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        qkv_bias: Whether to include bias in the QKV linear layers
        proj_bias: Whether to include bias in the attention output projection
    """
    def __init__(self, dim: int, head_dim: int = 64, qkv_bias: bool = False, proj_bias: bool = False):
        super().__init__()
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        # TODO: Define here the linear layer(s) producing K, Q, V from the input x
        # Hint: Do you need to define three different projections, or can you use a single one for all three?
            

        ## Begining of My code 
        self.qkv_proj = nn.Linear(dim, 3 * self.num_heads * head_dim, bias=qkv_bias)
        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.head_dim=head_dim # I believe this is more intersting to do, in order to avoid having to do the calculation in the forward method as it is directly inputed to the init function

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape # Batch size, sequence length, and dimension

        # TODO: Compute the keys K, queries Q, and values V from x. Each should be of shape [B num_heads L head_dim].
        qkv=self.qkv_proj(x) # B, L , 3*num_heads*head_dim 
        qkv= qkv.view(B, L, 3, self.num_heads, self.head_dim)# To cut the 3*num_heads*head_dim in to -> 3,num_head,head_dim  hence we have    (B,L,3,num_head,head_dim )
        qkv=rearrange(qkv, 'b l three h d -> three b h l d')#[3 ,B num_heads L head_dim]
        q, k, v = qkv[0],qkv[1],qkv[2] # [B, num_heads L head_dim]

        # TODO: Compute the attention matrix (pre softmax) and scale it by 1/sqrt(d_k). It should be of shape [B num_heads L L].
        # Hint: Use the already defined self.scale
        attn = (q@k.transpose(2,3))*self.scale #[B num_heads L L]

        if mask is not None:
            mask = rearrange(mask, "b n m -> b 1 n m") # Unsqueeze for multi-head attention
            # TODO: Apply the optional attention mask. Wherever the mask is False, replace the attention 
            # matrix value by negative infinity → zero attention weight after softmax.
            attn = attn.masked_fill_(~mask,float('-inf')) #[B num_heads L L] here i use the converse because i will use tril wich will put the one one the lower triangle of the matrix

        # TODO: Compute the softmax over the last dimension
        attn = torch.softmax(attn, dim=-1) #[B num_heads L L]

        # TODO: Weight the values V by the attention matrix and concatenate the different attention heads
        # Make sure to reshape the output to the original shape of x, i.e. [B L D]
        x = attn@v #[B num_heads L head_dim]

        x = rearrange(x, 'b h l d -> b l (h d)') #(B,L,self.num_heads*self.head_dim)

        # Output projection
        x = self.attn_out_proj(x)
        return x


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention module.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        qkv_bias: Whether to include bias in the QKV linear layers
        proj_bias: Whether to include bias in the attention output projection
    """
    def __init__(self, dim: int, head_dim: int = 64, qkv_bias: bool = False, proj_bias: bool = False):
        super().__init__()
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        # TODO: Define here the linear layer producing Q from the input x
        self.q=nn.Linear(dim,self.num_heads*head_dim,qkv_bias)
        
        # TODO: Define here the linear layers producing K, V from the context
        # Hint: Do you need to define two different projections, or can you use a single one for both?
        self.kv=nn.Linear(dim,2*self.num_heads*head_dim,qkv_bias)
        

        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.head_dim = head_dim #as i did for the first notebook i think this is better than having to recalculate it 

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape # Batch size, x sequence length (N), and dimension
        _, M, _ = context.shape # _, context sequence length (M), _

        # TODO: Compute the queries Q from x. It should be of shape [B num_heads N head_dim].
        q = self.q(x)                                            # [B, N, H*d]
        q = q.view(B, N, self.num_heads, self.head_dim)          # [B, N, H, d]
        q = q.permute(0, 2, 1, 3)                                # [B, H, N, d]
        
        # TODO: Compute the keys K and values V from the context. Each should be of shape [B num_heads M head_dim].
        kv = self.kv(context)                                    # [B, M, 2*H*d]
        kv = kv.view(B, M, 2, self.num_heads, self.head_dim)     # [B, M, 2, H, d]
        kv = kv.permute(2, 0, 3, 1, 4)                           # [2, B, H, M, d]
        k, v = kv[0], kv[1]                                      # each [B, H, M, d]

        
        

        # TODO: Compute the attention matrix (pre softmax) and scale it by 1/sqrt(d_k). It should be of shape [B num_heads N M].
        # Hint: Use the already defined self.scale
        attn =  (q@k.transpose(2,3))*self.scale

        if mask is not None:
            mask = rearrange(mask, "b n m -> b 1 n m") # Unsqueeze for multi-head attention
            # TODO: Apply the optional attention mask. Wherever the mask is False, replace the attention 
            # matrix value by negative infinity → zero attention weight after softmax.
            attn = attn.masked_fill_(~mask,float('-inf'))

        # TODO: Compute the softmax over the last dimension
        attn = torch.softmax(attn, dim=-1)

        # TODO: Weight the values V by the attention matrix and concatenate the different attention heads
        # Make sure to reshape the output to the original shape of x, i.e. [B N D]
        x = attn@v 
        x = rearrange(x, 'b h l d -> b l (h d)')
        # Output projection
        x = self.attn_out_proj(x)

        return x


class Block(nn.Module):
    """
    Basic transformer block with a multi-head self-attention mechanism and a feed-forward MLP.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(self, dim: int, head_dim: int = 64, mlp_ratio: float = 4., use_bias: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias=use_bias) # TODO (use the LayerNorm defined above)
        self.attn = Attention(dim, head_dim, qkv_bias=use_bias, proj_bias=use_bias) # TODO  
        #torch.triu(torch.ones(head_dim,head_dim)
        self.norm2 = LayerNorm(dim,bias=use_bias) # TODO (use the LayerNorm defined above)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim,mlp_hidden_dim,dim,use_bias) # TODO

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x= x + self.attn(self.norm1(x),mask) # TODO
        x= x + self.mlp(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    """
    Basic transformer decoder block with a multi-head self-attention, 
    a multi-head cross-attention, and a feed-forward MLP layer.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(self, dim: int, head_dim: int = 64, mlp_ratio: float = 4., use_bias: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(dim,bias=use_bias) # TODO (use the LayerNorm defined above)
        self.query_norm = LayerNorm(dim,bias=use_bias) # TODO (use the LayerNorm defined above)
        self.context_norm = LayerNorm(dim,bias=use_bias)# TODO (use the LayerNorm defined above)
        self.norm2 = LayerNorm(dim,bias=use_bias) # TODO (use the LayerNorm defined above)

        self.self_attn = Attention(dim,head_dim,qkv_bias=use_bias,proj_bias=use_bias) # TODO Attention layer
        self.cross_attn = CrossAttention(dim,head_dim,qkv_bias=use_bias,proj_bias=use_bias) # TODO CrossAttention layer

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim,mlp_hidden_dim,dim,use_bias) # TODO MLP layer

    def forward(self, 
            x: torch.Tensor, 
            context: torch.Tensor, 
            sa_mask: Optional[torch.Tensor] = None, # Self-attention mask
            xa_mask: Optional[torch.Tensor] = None, # Cross-attention mask
        ) -> torch.Tensor:

        # Self-attention, then cross-attention, then MLP
        # Make sure to apply the self-attention mask (sa_mask) to the self-attention layer,
        # and the cross-attention mask (xa_mask) to the cross-attention layer.
        # Don't forget to add the residual connections after each layer, and
        # to apply the normalizations on the inputs of each layer.
        x=x + self.self_attn(self.norm1(x),mask=sa_mask)
        x=x+self.cross_attn(self.query_norm(x),self.context_norm(context),mask=xa_mask)
        x=x+self.mlp(self.norm2(x))
        return x


class TransformerTrunk(nn.Module):
    """Basic Transformer trunk definition that can be used for encoder-only,
    decoder-only and prefixLM models, depending on the attention mask applied.

    Args:
        dim: Transformer dimension
        depth: Number of transformer layers
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(
        self,
            dim: int = 512,
            depth: int = 8,
            head_dim: int = 64,
            mlp_ratio: float = 4.0,
            use_bias: bool = False,
        ):
        super().__init__()

        self.blocks = nn.ModuleList([Block(dim,head_dim,mlp_ratio,use_bias) for i in range (depth)]) # TODO: Create a list of transformer blocks and wrap inside nn.ModuleList
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        
        for i, l in enumerate(self.blocks):
            x = l(x,mask=mask) 
        return x


class TransformerDecoderTrunk(nn.Module):
    """Basic Transformer decoder with interleaved self- and cross-attention, that can
    be used as the decoder for encoder-decoder models.

    Args:
        dim: Transformer dimension
        depth: Number of transformer layers
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(
        self,
            dim: int = 512,
            depth: int = 8,
            head_dim: int = 64,
            mlp_ratio: float = 4.0,
            use_bias: bool = False,
        ):
        super().__init__()

        self.blocks = nn.ModuleList([DecoderBlock(dim, head_dim, mlp_ratio, use_bias) for i in range (depth)]) # TODO: Create a list of transformer decoder blocks and wrap inside nn.ModuleList
    
    def forward(
            self, 
            x: torch.Tensor, 
            context: torch.Tensor, 
            sa_mask: Optional[torch.Tensor] = None, # Self-attention mask
            xa_mask: Optional[torch.Tensor] = None, # Cross-attention mask
        ) -> torch.Tensor:
        
        for i, l in enumerate(self.blocks):
            x = l(x, context, sa_mask=sa_mask, xa_mask=xa_mask)
        return x