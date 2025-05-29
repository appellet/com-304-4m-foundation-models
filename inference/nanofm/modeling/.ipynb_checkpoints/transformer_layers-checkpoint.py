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
        
        self.block = nn.Sequential (
            nn.Linear(in_features = in_features, out_features= hidden_features, bias = bias ), 
            nn.GELU(),
            nn.Linear(in_features = hidden_features , out_features= out_features, bias = bias)
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
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

     

        # the input features dimension is 'dim' and the output is 3 projections of the input (Q,K,V projections) of 'num_heads'
        # of 'head_dim' dimension. 
        # the objective is to have for each q,k,v we have 'num_heads' matrices of projection  of shape [  dim , head_dim ]
        # with head_dim = the dimensionality of the k,v,q projection output features.    
        self.projection_Q_K_V = nn.Linear(dim, 3 * self.num_heads * head_dim, bias=qkv_bias)
        
        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias) # after concatenation we make an output projection 


    def forward(self, x: torch.Tensor, mask : Optional[torch.Tensor] = None) -> torch.Tensor:
        print("Shape of x:", x.shape)
        B, L, D = x.shape  # Batch size, sequence length, and dimension
        
        head_dim = D // self.num_heads # compute the head_dim 

        #   X @ Wq.T ( (Batch size , sequence length,  dimension) * ( dimension, num_heads * head-dim * 3 ) -->
        #  ( Batch size, sequence length, num_heads *   head-dim *  3 ) )  --> ( B,L, num_heads * head-dim * 3 ) 
        #  ( B,L, num_heads * head-dim * 3 ) --> (3 ,B, head nmb, L, head-dim )
        #  (3 ,B, head nmb, L, head-dim ) --> [B num_heads L head_dim] for each q,k,v
        q,k,v = self.projection_Q_K_V(x).reshape(B, L, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        

        # Compute the attention matrix (pre softmax) and scale it by 1/sqrt(d_k). It should be of shape [B num_heads L L].

        # computation of the similarity matrix  q : [B num_heads L head_dim] @ k.T : [B num_heads  head_dim L ] -->  [B num_heads L L]
        attn = (q@k.transpose(-1,-2)) * self.scale
        
        if mask is not None:
                mask = rearrange(mask, "b n m -> b 1 n m") # Unsqueeze for multi-head attention
            
            # Apply the optional attention mask. Wherever the mask is False, replace the attention 
            # matrix value by negative infinity → zero attention weight after softmax.

            # we use the 0 values (False) positions in our mask to make the values with the same positions in attn equal to infinity
                attn = attn.masked_fill_(~mask,float('-inf')) 

        # TODO: Compute the softmax over the last dimension
        # for each query in the similarity matrix (row) we compute softmax over the keys (colums (last dimension)) 
        attn = torch.softmax(attn,dim=-1)  # attn :  [B num_heads L L]


        # TODO: Weight the values V by the attention matrix and concatenate the different attention heads
        # Make sure to reshape the output to the original shape of x, i.e. [B L D]
        # attn :  [B num_heads L L]    v : [B num_heads L head_dim] ( for each value feature and each query we will weight the value
        # of token with the similarity of keys to the query (attention)) output : [ B num_heads L head_dim] then we reshape to concat
        # and have X :  [B,L,D] with D = num_heads * head_dim 
        x = attn@v
        x = rearrange(x, 'b h l d -> b l (h d)')

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
        self.projection_layer_Q = nn.Linear(dim, self.num_heads * head_dim, bias=qkv_bias)
        
        # TODO: Define here the linear layers producing K, V from the context
        # Hint: Do you need to define two different projections, or can you use a single one for both?
        self.projection_layer_K_V = nn.Linear(dim , self.num_heads * head_dim * 2)

        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape # Batch size, x sequence length (N), and dimension
        _, M, _ = context.shape # _, context sequence length (M), _

        head_dim = C // self.num_heads
        
        # TODO: Compute the queries Q from x. It should be of shape [B num_heads N head_dim].
        q = self.projection_layer_Q(x).reshape(B, N , self.num_heads, head_dim).permute(0, 2, 1, 3)

        # TODO: Compute the keys K and values V from the context. Each should be of shape [B num_heads M head_dim].
        k,v = self.projection_layer_K_V(context).reshape(B, M, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)

        # TODO: Compute the attention matrix (pre softmax) and scale it by 1/sqrt(d_k). It should be of shape [B num_heads N M].
        # Hint: Use the already defined self.scale
        attn = (q@k.transpose(-1,-2)) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b n m -> b 1 n m") # Unsqueeze for multi-head attention
            # TODO: Apply the optional attention mask. Wherever the mask is False, replace the attention 
            # matrix value by negative infinity → zero attention weight after softmax.
            attn = attn.masked_fill_(~mask,float('-inf')) 

        # TODO: Compute the softmax over the last dimension
        attn = torch.softmax(attn,dim=-1)

        # TODO: Weight the values V by the attention matrix and concatenate the different attention heads
        # Make sure to reshape the output to the original shape of x, i.e. [B N D]
        # for each query (a query per token) we have M similarity scores with keys (from context )and so with M values (from context) 
        # we iterate on the features  of the values and weight each value of the this feature with the similarity score corresponding  
        # ==> for each token (query) we have summed the weighted values for each token dimension feature. 
        x = attn@v
        x = rearrange(x, 'b h n d -> b n (h d)')
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
        self.norm1 = LayerNorm(dim, bias = use_bias )  
        self.attn =  Attention(dim, head_dim, qkv_bias= use_bias, proj_bias= use_bias) 
        self.norm2 =  LayerNorm(dim, bias = use_bias )  
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp =  Mlp(in_features = dim , hidden_features = mlp_hidden_dim , out_features = dim , bias = use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x= x + self.attn(self.norm1(x),mask) 
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
        self.norm1 = LayerNorm(dim, bias = use_bias ) # TODO (use the LayerNorm defined above)
        self.query_norm =  LayerNorm(dim, bias = use_bias ) # TODO (use the LayerNorm defined above)
        self.context_norm =  LayerNorm(dim, bias = use_bias ) # TODO (use the LayerNorm defined above)
        self.norm2 = LayerNorm(dim, bias = use_bias) # TODO (use the LayerNorm defined above)

        self.self_attn = Attention(dim, head_dim, qkv_bias= use_bias, proj_bias= use_bias)  # TODO Attention layer
        self.cross_attn = CrossAttention(dim, head_dim, qkv_bias = use_bias, proj_bias = use_bias) # TODO CrossAttention layer

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features = dim , hidden_features = mlp_hidden_dim , out_features = dim , bias = use_bias) # TODO MLP layer

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
        x = x + self.self_attn(self.norm1(x), sa_mask ) # TODO
        x = x + self.cross_attn(self.query_norm(x),self.context_norm(context), xa_mask)
        x = x + self.mlp(self.norm2(x))
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
        self.blocks =  nn.ModuleList([Block(dim= dim, head_dim = head_dim, mlp_ratio = mlp_ratio, use_bias= use_bias ) for i in range (depth)])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x,mask=mask) 
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

        self.blocks = nn.ModuleList([DecoderBlock(dim, head_dim, mlp_ratio, use_bias) for i in range(depth)])
        # TODO: Create a list of transformer decoder blocks and wrap inside nn.ModuleList
    
    def forward(
            self, 
            x: torch.Tensor, 
            context: torch.Tensor, 
            sa_mask: Optional[torch.Tensor] = None, # Self-attention mask
            xa_mask: Optional[torch.Tensor] = None, # Cross-attention mask
        ) -> torch.Tensor:
        
        for block in self.blocks:
            x = block(x,context, sa_mask = sa_mask , xa_mask = xa_mask) 
        return x

