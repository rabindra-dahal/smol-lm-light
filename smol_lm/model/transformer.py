import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import RMSNorm
from .attention import SmolGQAAttention
from .feedforward import SwiGLUFeedForward

class SmolBlock(nn.Module):
    """
    A single complete Transformer layer integrating Attention, SwiGLU MLP, and absolute sequence metrics.
    
    ===========================================================================
    MATHEMATICS INVOLVED (Residual Connections Framework):
    To stabilize gradient pipelines during deep training loops, pre-norm routing is applied:
        
        x_1   = x + Attention( RMSNorm(x), position_offset )
        x_out = x_1 + SwiGLU( RMSNorm(x_1) )
    ===========================================================================
    
    WHY IT IS USED:
    Acting as an architectural bridge, this block handles input/output dimensions. 
    By passing the `position_offset` variable directly from the top model call down into the 
    underlying `SmolGQAAttention` layer, it ensures absolute word-indexing properties remain intact 
    through every residual addition step.
    """
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff, max_seq_len):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = SmolGQAAttention(d_model, num_heads, num_kv_heads, max_seq_len)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff)

    def forward(self, x, position_offset=0): 
        # Pass rolling position index offsets straight into GQA attention engine forward calls
        x = x + self.attn(self.attn_norm(x), position_offset=position_offset) 
        x = x + self.ffn(self.ffn_norm(x))
        return x

class SmolLM(nn.Module):
    """
    The unified multi-layer neural network backbone framework blueprint of SmolLM.
    
    ===========================================================================
    MATHEMATICS INVOLVED (Orchestration Pipeline Workflow):
        1. Token IDs Tensor Matrix -> Embedding Matrix Vector Coordinate Space.
        2. Hidden Tensors Sequence Map -> Iterated across N deep structural SmolBlocks.
        3. Terminus Vector Arrays -> Final RMSNorm Calibration -> Linear Vocabulary Head Decoded.
        
    Weight Tying Identity Matrix Equation Binding:
        Embedding_Matrix.weight = Output_Decoding_Head.weight
    ===========================================================================
    
    WHY IT IS USED:
    This class ties your model's individual files into a clean project package namespace. 
    By adding the explicit `position_offset` structural parameter handling arguments directly 
    into the top level `forward` declaration loop signature, it allows your training script 
    and your generation script to pass real-time context metrics into the model smoothly.
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, num_kv_heads, d_ff, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Compile multi-block layers into structural modules stack arrays
        self.layers = nn.ModuleList([
            SmolBlock(d_model, num_heads, num_kv_heads, d_ff, max_seq_len) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Reuses input token embedding weight space inside final linear head decoding arrays to trim RAM allocation
        self.token_embedding.weight = self.output_head.weight 
        
    def forward(self, idx, position_offset=0):
        # 1. Translate numeric token indices directly into coordinate tracking vectors
        x = self.token_embedding(idx) 
        
        # 2. Loop sequences through all stacked transformer layers, maintaining absolute index offset metrics
        for layer in self.layers: 
            x = layer(x, position_offset=position_offset)
            
        # 3. Apply layer norm and decode back to vocabulary logit space [Batch, Sequence, Vocab_Size]
        return self.output_head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressive text generation inline preview tracker used during dataset training checkpoints."""
        self.eval() # Disable network learning metrics updates tracking
        for _ in range(max_new_tokens):
            # Inline loops scale relative from position index 0 across max context parameters boundaries
            logits = self(idx[:, -self.max_seq_len:], position_offset=0)[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        self.train() # Flip parameters back to model optimization settings state
        return idx.tolist()
