import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniTransformerBlock(nn.Module):
    """
    Mini Transformer Block implemented with PyTorch.
    
    This block consists of:
    1. Multi-head self-attention with residual connection and layer normalization
    2. Feed-forward network (2-layer MLP) with residual connection and layer normalization
    
    Input shape: (Batch_size, Seq_len, d_model)
    Output shape: (Batch_size, Seq_len, d_model)
    """
    
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Initialize the Mini Transformer Block.
        
        Args:
            d_model: Dimension of the model (embedding dimension)
            n_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network (default: 2048)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # Input shape: (batch, seq_len, d_model)
        )
        
        # Feed-forward network (2-layer MLP)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Mini Transformer Block.
        
        Args:
            x: Input tensor of shape (Batch_size, Seq_len, d_model)
            attn_mask: Optional attention mask of shape (Seq_len, Seq_len) or (Batch_size * n_heads, Seq_len, Seq_len)
            
        Returns:
            Output tensor of shape (Batch_size, Seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, f"Input d_model ({d_model}) doesn't match block d_model ({self.d_model})"
        
        # ==================== Self-Attention Sub-layer ====================
        # Save residual connection
        residual = x  # Shape: (Batch_size, Seq_len, d_model)
        
        # Layer normalization before attention
        x_norm = self.norm1(x)  # Shape: (Batch_size, Seq_len, d_model)
        
        # Self-attention
        # attn_output: (Batch_size, Seq_len, d_model)
        # attn_weights: (Batch_size, Seq_len, Seq_len) or None
        attn_output, attn_weights = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=attn_mask,
            need_weights=False  # Set to True if you need attention weights
        )
        
        # Apply dropout to attention output
        attn_output = self.dropout1(attn_output)  # Shape: (Batch_size, Seq_len, d_model)
        
        # Residual connection
        x = residual + attn_output  # Shape: (Batch_size, Seq_len, d_model)
        
        # ==================== Feed-Forward Sub-layer ====================
        # Save residual connection
        residual = x  # Shape: (Batch_size, Seq_len, d_model)
        
        # Layer normalization before FFN
        x_norm = self.norm2(x)  # Shape: (Batch_size, Seq_len, d_model)
        
        # Feed-forward network
        ffn_output = self.ffn(x_norm)  # Shape: (Batch_size, Seq_len, d_model)
        
        # Apply dropout to FFN output
        ffn_output = self.dropout2(ffn_output)  # Shape: (Batch_size, Seq_len, d_model)
        
        # Residual connection
        x = residual + ffn_output  # Shape: (Batch_size, Seq_len, d_model)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Get attention weights from the self-attention layer.
        
        Args:
            x: Input tensor of shape (Batch_size, Seq_len, d_model)
            attn_mask: Optional attention mask
            
        Returns:
            Attention weights of shape (Batch_size, n_heads, Seq_len, Seq_len) or (Batch_size, Seq_len, Seq_len)
        """
        x_norm = self.norm1(x)
        
        # Get attention weights
        _, attn_weights = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False  # Return attention weights per head
        )
        
        return attn_weights


def test_mini_transformer_block():
    """Test function to verify the MiniTransformerBlock implementation.
    
    Parameters are optimized for RTX 4050 6GB to avoid OOM (Out Of Memory) errors.
    """
    import numpy as np
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test parameters optimized for RTX 4050 6GB
    # Reduced to avoid OOM while maintaining model capabilities
    batch_size = 2          # Reduced from 4 to minimize batch memory
    seq_len = 32           # Increased to test longer sequences
    d_model = 256          # Reduced from 512 to lower memory footprint
    n_heads = 4            # Reduced from 8, must divide d_model evenly
    dim_feedforward = 1024 # Reduced from 2048 for FFN layer
    
    # Verify d_model is divisible by n_heads
    assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    
    print("Testing MiniTransformerBlock (optimized for RTX 4050 6GB)...")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}, d_model: {d_model}, n_heads: {n_heads}")
    print(f"Feedforward dimension: {dim_feedforward}")
    
    # Check GPU memory if available
    if torch.cuda.is_available():
        print(f"\nCUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    # Create the block
    block = MiniTransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        dropout=0.1
    )
    
        # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = block.to(device)
    print(f"\nModel moved to: {device}")
    
    # Monitor GPU memory before creating input
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU memory allocated after model: {memory_allocated:.2f} GB")
        print(f"GPU memory reserved after model: {memory_reserved:.2f} GB")
    
    # Create random input on the same device as the model
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    print(f"\nInput shape: {x.shape} (on {device})")
    
    # Forward pass
    output = block(x)
    print(f"Output shape: {output.shape} (on {device})")
    
    # Verify input and output shapes match
    assert x.shape == output.shape, f"Shape mismatch: input {x.shape} != output {output.shape}"
    print("✓ Input and output shapes match")
    
        # Test with attention mask
    attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # Move mask to same device as model
    attn_mask = attn_mask.to(device)
    output_masked = block(x, attn_mask=attn_mask)
    print(f"Output with mask shape: {output_masked.shape}")
    print("✓ Masked attention works")
    
        # Test attention weights
    attn_weights = block.get_attention_weights(x)
    if attn_weights.dim() == 4:
        print(f"Attention weights shape (per head): {attn_weights.shape}")
    else:
        print(f"Attention weights shape (averaged): {attn_weights.shape}")
    print("✓ Attention weights can be retrieved")
    
    # Monitor GPU memory after all operations
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"\nGPU memory allocated after all ops: {memory_allocated:.2f} GB")
        print(f"GPU memory reserved after all ops: {memory_reserved:.2f} GB")
        
        # Check if we're using a reasonable amount of memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        memory_usage_percent = (memory_allocated / total_memory) * 100
        print(f"Memory usage: {memory_usage_percent:.1f}% of total GPU memory")
        
        if memory_allocated > 4.0:  # More than 4GB used
            print("⚠️  Warning: High GPU memory usage detected!")
            print("   Consider further reducing batch_size, seq_len, or d_model.")
        else:
            print("✓ GPU memory usage is within safe limits for RTX 4050 6GB")
    
        # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Calculate memory estimate for parameters (assuming float32)
    param_memory_mb = (total_params * 4) / (1024 ** 2)  # 4 bytes per float32
    print(f"Estimated parameter memory: {param_memory_mb:.2f} MB")
    
    print("\n✅ All tests passed! MiniTransformerBlock is working correctly.")
    print("\n📊 Summary of optimized parameters for RTX 4050 6GB:")
    print(f"   • Batch size: {batch_size} (small batches reduce memory)")
    print(f"   • Sequence length: {seq_len} (balanced for testing)")
    print(f"   • Model dimension: {d_model} (reduced from 512)")
    print(f"   • Attention heads: {n_heads} (reduced from 8)")
    print(f"   • Feedforward dim: {dim_feedforward} (reduced from 2048)")
    print("\n💡 If you still encounter OOM errors, try:")
    print("   1. Reduce batch_size to 1")
    print("   2. Reduce seq_len to 16")
    print("   3. Reduce d_model to 128")
    print("   4. Use mixed precision (torch.cuda.amp) for training")
    
    return block, x, output


if __name__ == "__main__":
    # Run tests when the file is executed directly
    block, x, output = test_mini_transformer_block()