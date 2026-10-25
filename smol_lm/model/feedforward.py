import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFeedForward(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU) Layer.
    
    ===========================================================================
    MATHEMATICS INVOLVED:
    SwiGLU operates two simultaneous hidden dense matrix parameter steps. 
    One acts as a non-linear activation filter gate, modulating the data from the other.
    
    SwiGLU(x) = ( SiLU( x @ W_1 ) * ( x @ W_2 ) ) @ W_3
    
    Where:
    SiLU(z) = z * Sigmoid(z) = z / ( 1 + e^(-z) )
    '*'     = Element-wise (Hadamard) matrix multiplication operator.
    ===========================================================================
    
    WHY IT IS USED:
    Classic Multi-Layer Perceptrons rely on standard single-matrix transformations 
    (like plain ReLU or GELU). SwiGLU splits the layer into a dual-matrix channel pipeline. 
    The smooth gating behavior of the continuous SiLU activation function eliminates 
    dead-neuron gradient stagnation points, allowing deep language networks to converge 
    much faster and lock onto complex data patterns efficiently.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False) # Gating network projection layer matrix
        self.w2 = nn.Linear(d_model, d_ff, bias=False) # Context text representation projection layer matrix
        self.w3 = nn.Linear(d_ff, d_model, bias=False) # External down-sampling recovery layer matrix

    def forward(self, x): 
        # Math: Apply element-wise multiplication between SiLU-activated w1 and linear w2 outputs, then project back out
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
