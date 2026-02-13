import torch

def apply_rotary_pos_emb(x, seq_dim=-2):
    """
    x: Input tensor
    seq_dim: The dimension that corresponds to sequence length. 
             Usually -2 for [Batch, Heads, Seq, Dim]
    """
    # 1. FIX PRECISION: Force calculations in float32
    n_dim = x.shape[-1]
    seq_len = x.shape[seq_dim]
    device = x.device
    
    # 2. FIX PAIRING: Use 'Sliced' strategy (Standard LLaMA/HF style)
    # Pairs are (0, dim/2), (1, dim/2 + 1)... not (0, 1)
    # We compute theta for half the dimensions
    theta = 1.0 / (10000 ** (torch.arange(0, n_dim, 2, device=device).float() / n_dim))
    
    # 3. FIX SHAPE TRAP: Ensure we use the correct sequence length
    timestamps = torch.arange(seq_len, device=device).float()
    
    # Create Angle Matrix [SeqLen, Dim/2]
    # outer product logic
    freqs = torch.outer(timestamps, theta) # Shape: [Seq, Dim/2]
    
    # Expand to full [Seq, Dim] by concatenating (matching the sliced pairing)
    # This aligns with [x_first_half, x_second_half]
    angle = torch.cat((freqs, freqs), dim=-1) # Shape: [Seq, Dim]
    
    # 4. FIX BROADCASTING: Reshape angle to align with x
    # If x is [B, H, S, D], angle needs to be [1, 1, S, D]
    # We construct a view that matches x's dimensions
    shape = [1] * x.ndim
    shape[seq_dim] = seq_len
    shape[-1] = n_dim
    angle = angle.view(*shape)
    
    # Cast back to input dtype (fp16/bf16) before applying sin/cos
    cos = angle.cos().to(x.dtype)
    sin = angle.sin().to(x.dtype)
    
    # 5. ROTATE (Sliced Version)
    def rotate_half(t):
        # Split into two halves (standard LLaMA)
        x1, x2 = t[..., : n_dim // 2], t[..., n_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    return (x * cos) + (rotate_half(x) * sin)