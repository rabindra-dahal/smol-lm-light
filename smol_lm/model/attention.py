import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import RoPEEmbedding

class SmolGQAAttention(nn.Module):
    r"""
    Grouped-Query Attention (GQA) Engine Layer with absolute position sequence tracking.
    
    ================================================================================================
    MATHEMATICS INVOLVED:
    1. Grouped-Query Scaling Matrix Interleave Ratio:
       g = num_heads / num_kv_heads
       
    2. Scaled Dot-Product Formulation:
       Scores = (Q @ K^T) / sqrt(head_dim)
       Matrix Dimension Shapes: [B, H, T_q, d_head] @ [B, H, d_head, T_k] -> [B, H, T_q, T_k]
       
    3. 4D Vector Broadcasting Mechanics:
       cos_sliced shape: [1, T, 1, d_head // 2] matches the exact dimensional rank of q_even [B, T, H, d//2]
       allowing PyTorch to multiply tracking tokens along Axis 1 while copying configurations over Axis 0 and 2.
    ================================================================================================
    
    WHY IT IS USED:
    1. GQA: Cuts down the memory footprint of Key/Value activations to run efficiently on laptop CPUs.
    2. Precise Index Slicing: By querying the extended RoPE cache directly using `[position_offset : position_offset + T]`, 
       the layer extracts exactly 'T' positional wave rows even when the sliding context window pushes past 
       the original dataset block bounds. This guarantees that dimension shapes remain perfectly aligned 
       throughout long autoregressive inference loops.
    """
    def __init__(self, d_model, num_heads, num_kv_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        # Linear projections (No-bias standard setups matching Llama architecture constraints)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Instantiate local RoPE coordinate cached grid structures
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len)
        
    def forward(self, x, position_offset=0):
        B, T, C = x.shape # Batch size (B), Sequence Length (T), Channel Dimensions (C)
        
        # 1. Project inputs into intermediate multi-head representation states
        # Shape: [Batch, Sequence_Len (T), Head_Count, Head_Dim]
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 2. --- CORRECTED 4D ROPE APPLICATION WITH SAFECATCH SLICING ---
        # Fetch the complete extended positional wave matrix arrays from the updated embedding class
        raw_cos, raw_sin = self.rope(q, k, position_offset + T)
        
        # Slice waves precisely using the absolute index offset tracker coordinates.
        # Adding unsqueeze(0) and unsqueeze(2) expands the matrix shape: [T, d//2] -> [1, T, 1, d//2]
        cos = raw_cos[position_offset : position_offset + T].unsqueeze(0).unsqueeze(2)
        sin = raw_sin[position_offset : position_offset + T].unsqueeze(0).unsqueeze(2)
        
        # Segment target matrices into alternating Even [0, 2, 4...] and Odd [1, 3, 5...] index spaces
        q_even, q_odd = q[..., ::2], q[..., 1::2]
        k_even, k_odd = k[..., ::2], k[..., 1::2]
        
        # Math: Compute vector plane rotations before transposing the sequence layout
        # Broadcast multiplication now maps flawlessly across [B, T, H, d_head//2] without shape crashes
        q = torch.cat([q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], dim=-1)
        k = torch.cat([k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], dim=-1)
        
        # Transpose matrices to structural heads attention shape layout: [Batch, Head_Count, Sequence_Len, Head_Dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        
        # 3. --- GQA MULTI-HEAD INTERLEAVE REPETITION ---
        # If KV heads count < Q heads count, duplicate elements along the group dimension to balance matrices
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # 4. Math: Scaled Dot-Product Similarity Scoring Equation: S = (Q @ K^T) / sqrt(d_head)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 5. --- DYNAMIC CAUSAL RECTANGULAR MASK MECHANIC ---
        # Generate row indices (i) and column indices (j) dynamically matching current matrix shapes
        rows = torch.arange(T, device=x.device).view(-1, 1) # [T, 1]
        cols = torch.arange(T, device=x.device).view(1, -1) # [1, T]
        
        # Math Causal Verification: Column position index must be <= Row position index
        causal_mask = (cols > rows).to(device=x.device)
        
        # Inject negative infinity (-\infty) into lookahead positions to completely block attention leakage
        mask = torch.zeros((T, T), device=x.device)
        mask = mask.masked_fill(causal_mask, float('-inf'))
        
        # Apply mask and execute softmax scaling distributions
        attn_weights = F.softmax(scores + mask, dim=-1)
        
        # 6. Math: Final Context Matrix Output Extraction: Context = Attention_Weights @ V
        context = attn_weights @ v
        
        # Flatten multi-head channels back cleanly into original unified 3D sequence hidden model dimensions
        return self.out_proj(context.transpose(1, 2).contiguous().view(B, T, C))
