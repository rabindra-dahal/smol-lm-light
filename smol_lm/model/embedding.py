import torch
import torch.nn as nn

class RoPEEmbedding(nn.Module):
    r"""
    Rotary Position Embeddings (RoPE) with Extended Cache allocation boundaries.
    
    ================================================================================================
    MATHEMATICS INVOLVED:
    For a target hidden query representation vector dimension 'dim' (which represents d_head):
        
        \theta_i = 10000^( -2(i-1) / dim )  for i in [1, 2, ..., dim // 2]
        
    We construct an extended continuous position index timeline array:
        t = [0, 1, 2, ..., (max_seq_len * 8) - 1]
        
    Geometric Angle Wave Matrix computation via Outer Vector Product:
        M_{m, i} = t_m * \theta_i
        Yielding a cached grid coordinate topology shaped: [(max_seq_len * 8), dim // 2]
    ================================================================================================
    
    WHY IT IS USED:
    During text inference generation, the token array grows continuously. If the position wave cache 
    matrix is tightly capped at `max_seq_len` (e.g., 32 rows), then as soon as the sequence shifts 
    forward into row step 33, the attention layer attempts to query index positions past the matrix boundary. 
    This triggers a shape truncation down to 31 rows, leading to a critical PyTorch broadcast crash. 
    Multiplying the timeline allocation range by 8 acts as a safe buffer, providing plenty of absolute 
    position rows for long token decoding runs without re-allocating memory on the CPU.
    """
    def __init__(self, dim, max_seq_len):
        super().__init__()
        # Math: Generate constant geometric rotation frequencies correctly across the head dimension depth:
        # \theta_i = 10000^(-2i / dim) where the arange size matches dim exactly.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        
        # FIX: Pre-allocate an extended position timeline grid to avoid context boundary shape drops
        extended_max_len = max_seq_len * 8
        t = torch.arange(extended_max_len, dtype=torch.float32)
        
        # Math: Outer product vector multiplication yielding final angle weights matrix mapping: (t * \theta_i)
        # Resulting shape footprints evaluate exactly to: [Extended_Max_Len, Dim // 2]
        freqs = torch.outer(t, inv_freq)
        
        # Cache cosine and sine waves locally on memory structures to optimize runtime processing loops
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, q, k, seq_len):
        # Always return the full pre-cached global matrix wave vectors.
        # This completely decouples sequence generation step states from raw allocation constraints.
        return self.cos, self.sin
