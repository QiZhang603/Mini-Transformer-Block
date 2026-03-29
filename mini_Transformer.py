import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler


class PositionalEncoding(nn.Module):
    """Sinusoidal or learnable positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, learnable: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.learnable = learnable
        self.d_model = d_model

        if learnable:
            self.pos_embed = nn.Embedding(max_len, d_model)
        else:
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        if self.learnable:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
            pos = self.pos_embed(positions)
        else:
            pos = self.pe[:, : x.size(1)]
        x = x + pos
        return self.dropout(x)


def make_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Return key_padding_mask (batch, seq_len) with True at padding positions."""
    return tokens.eq(pad_id)


def make_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Upper-triangular mask with True where future tokens should be masked."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class MiniTransformerDecoderBlock(nn.Module):
    """Pre-LN Transformer decoder block with self-attn, cross-attn, and FFN."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention (masked)
        residual = tgt
        tgt_norm = self.norm1(tgt)
        attn_output, _ = self.self_attn(
            tgt_norm,
            tgt_norm,
            tgt_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = residual + self.dropout1(attn_output)

        # Cross-attention
        residual = tgt
        tgt_norm = self.norm2(tgt)
        cross_output, _ = self.cross_attn(
            tgt_norm,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        tgt = residual + self.dropout2(cross_output)

        # Feed-forward
        residual = tgt
        tgt_norm = self.norm3(tgt)
        ffn_output = self.ffn(tgt_norm)
        tgt = residual + self.dropout3(ffn_output)
        return tgt

    def get_attention_weights(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt_norm = self.norm1(tgt)
        _, self_weights = self.self_attn(
            tgt_norm,
            tgt_norm,
            tgt_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        tgt_norm = self.norm2(tgt)
        _, cross_weights = self.cross_attn(
            tgt_norm,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        return self_weights, cross_weights


class MiniTransformerDecoder(nn.Module):
    """Stacked decoder using MiniTransformerDecoderBlock."""

    def __init__(self, d_model: int, n_heads: int, num_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [
                MiniTransformerDecoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return tgt

    def get_attention_weights(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[list, list]:
        self_weights = []
        cross_weights = []
        current_tgt = tgt
        for layer in self.layers:
            sw, cw = layer.get_attention_weights(
                current_tgt,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            self_weights.append(sw)
            cross_weights.append(cw)
            current_tgt = layer(
                current_tgt,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self_weights, cross_weights


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
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Mini Transformer Block.
        
        Args:
            x: Input tensor of shape (Batch_size, Seq_len, d_model)
            attn_mask: Optional attention mask of shape (Seq_len, Seq_len) or (Batch_size * n_heads, Seq_len, Seq_len)
            key_padding_mask: Optional bool mask (Batch_size, Seq_len) with True at padding positions
            
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
            key_padding_mask=key_padding_mask,
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
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get attention weights from the self-attention layer.
        
        Args:
            x: Input tensor of shape (Batch_size, Seq_len, d_model)
            attn_mask: Optional attention mask
            key_padding_mask: Optional bool mask (Batch_size, Seq_len)
            
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
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # Return attention weights per head
        )
        
        return attn_weights


class MiniTransformerEncoder(nn.Module):
    """
    Mini Transformer Encoder implemented by stacking multiple MiniTransformerBlocks.
    
    This encoder consists of N stacked transformer blocks, where each block contains:
    1. Multi-head self-attention with residual connection and layer normalization
    2. Feed-forward network with residual connection and layer normalization
    
    Input shape: (Batch_size, Seq_len, d_model)
    Output shape: (Batch_size, Seq_len, d_model)
    """
    
    def __init__(self, d_model: int, n_heads: int, num_layers: int = 3, 
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Initialize the Mini Transformer Encoder.
        
        Args:
            d_model: Dimension of the model (embedding dimension)
            n_heads: Number of attention heads
            num_layers: Number of transformer blocks to stack (default: 3)
            dim_feedforward: Dimension of feedforward network (default: 2048)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        
        # Create a list of transformer blocks
        self.layers = nn.ModuleList([
            MiniTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Mini Transformer Encoder.
        
        Args:
            x: Input tensor of shape (Batch_size, Seq_len, d_model)
            attn_mask: Optional attention mask of shape (Seq_len, Seq_len) or (Batch_size * n_heads, Seq_len, Seq_len)
            key_padding_mask: Optional bool mask (Batch_size, Seq_len) with True at padding positions
            
        Returns:
            Output tensor of shape (Batch_size, Seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, f"Input d_model ({d_model}) doesn't match encoder d_model ({self.d_model})"
        
        # Pass through each transformer block
        for i, layer in enumerate(self.layers):
            # Shape remains (Batch_size, Seq_len, d_model) through each layer
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            
        return x
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> list:
        """
        Get attention weights from all layers of the encoder.
        
        Args:
            x: Input tensor of shape (Batch_size, Seq_len, d_model)
            attn_mask: Optional attention mask
            key_padding_mask: Optional bool mask (Batch_size, Seq_len)
            
        Returns:
            List of attention weights from each layer
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, f"Input d_model ({d_model}) doesn't match encoder d_model ({self.d_model})"
        
        attention_weights = []
        current_x = x
        
        # Get attention weights from each layer
        for layer in self.layers:
            # Get attention weights for current layer
            attn_weights = layer.get_attention_weights(current_x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            attention_weights.append(attn_weights)
            
            # Pass through the layer to get input for next layer
            current_x = layer(current_x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            
        return attention_weights


class MiniTransformer(nn.Module):
    """Encoder-Decoder Transformer with embeddings, positional encoding, and output head."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_heads: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
        learned_positional: bool = False,
        tie_embeddings: bool = False,
        share_encoder_decoder_embedding: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = self.src_embed if share_encoder_decoder_embedding else nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len, learnable=learned_positional)

        self.encoder = MiniTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = MiniTransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        if tie_embeddings:
            if tgt_vocab_size != src_vocab_size:
                raise ValueError("Tying embeddings requires src and tgt vocab sizes to match.")
            self.output_proj.weight = self.tgt_embed.weight

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        pad_id: Optional[int] = None,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, list]]]:
        pad_id = self.pad_id if pad_id is None else pad_id

        if src_key_padding_mask is None and pad_id is not None:
            src_key_padding_mask = make_padding_mask(src_tokens, pad_id)
        if tgt_key_padding_mask is None and pad_id is not None:
            tgt_key_padding_mask = make_padding_mask(tgt_tokens, pad_id)
        if tgt_mask is None:
            tgt_mask = make_causal_mask(tgt_tokens.size(1), device=tgt_tokens.device)

        src_embed = self.positional_encoding(self.src_embed(src_tokens) * math.sqrt(self.d_model))
        tgt_embed = self.positional_encoding(self.tgt_embed(tgt_tokens) * math.sqrt(self.d_model))

        memory = self.encoder(src_embed, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        decoder_out = self.decoder(
            tgt_embed,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = self.output_proj(decoder_out)

        if not return_attn:
            return logits

        enc_attn = self.encoder.get_attention_weights(src_embed, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        dec_self_attn, dec_cross_attn = self.decoder.get_attention_weights(
            tgt_embed,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        attn = {"encoder": enc_attn, "decoder_self": dec_self_attn, "decoder_cross": dec_cross_attn}
        return logits, attn


def build_training_components(model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.01, pad_id: int = 0):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
    return criterion, optimizer, scheduler


def training_step(
    model: MiniTransformer,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Optional[_LRScheduler] = None,
    clip_grad: float = 1.0,
) -> float:
    model.train()
    optimizer.zero_grad()

    src = batch["src"]
    tgt_in = batch["tgt_in"]
    tgt_out = batch["tgt_out"]

    logits = model(src_tokens=src, tgt_tokens=tgt_in)
    loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
    loss.backward()
    if clip_grad is not None:
        clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return float(loss.detach().cpu())


def test_full_transformer_forward():
    """Lightweight shape check for full Transformer forward pass."""
    torch.manual_seed(0)

    batch_size = 2
    src_len = 6
    tgt_len = 5
    vocab_size = 32
    pad_id = 0

    src = torch.randint(1, vocab_size, (batch_size, src_len))
    tgt_in = torch.randint(1, vocab_size, (batch_size, tgt_len))
    tgt_out = torch.randint(1, vocab_size, (batch_size, tgt_len))

    model = MiniTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=64,
        pad_id=pad_id,
    )

    logits = model(src_tokens=src, tgt_tokens=tgt_in, pad_id=pad_id)
    assert logits.shape == (batch_size, tgt_len, vocab_size), "Logits shape mismatch"
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    loss = loss_fn(logits.view(-1, vocab_size), tgt_out.view(-1))
    print(f"Full Transformer forward OK. Loss (random data): {loss.item():.4f}")


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


def test_mini_transformer_encoder():
    """Test function to verify the MiniTransformerEncoder implementation.
    
    Parameters are further optimized for RTX 4050 6GB to avoid OOM with multiple layers.
    """
    import numpy as np
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test parameters optimized for RTX 4050 6GB with 3 layers
    # Further reduced to avoid OOM with multiple stacked blocks
    batch_size = 1          # Reduced to 1 to minimize memory with 3 layers
    seq_len = 24           # Slightly reduced for safety
    d_model = 256          # Same as block test
    n_heads = 4            # Same as block test
    num_layers = 3         # Number of stacked blocks
    dim_feedforward = 1024 # Same as block test
    
    # Verify d_model is divisible by n_heads
    assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    
    print("\n" + "="*60)
    print("Testing MiniTransformerEncoder (optimized for RTX 4050 6GB)...")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}, d_model: {d_model}, n_heads: {n_heads}")
    print(f"Number of layers: {num_layers}, Feedforward dimension: {dim_feedforward}")
    
    # Check GPU memory if available
    if torch.cuda.is_available():
        print(f"\nCUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    # Create the encoder
    encoder = MiniTransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.1
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    print(f"\nEncoder moved to: {device}")
    
    # Monitor GPU memory before creating input
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU memory allocated after encoder: {memory_allocated:.2f} GB")
        print(f"GPU memory reserved after encoder: {memory_reserved:.2f} GB")
    
    # Create random input on the same device as the model
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    print(f"\nInput shape: {x.shape} (on {device})")
    
    # Forward pass through encoder
    output = encoder(x)
    print(f"Encoder output shape: {output.shape} (on {device})")
    
    # Verify input and output shapes match
    assert x.shape == output.shape, f"Shape mismatch: input {x.shape} != output {output.shape}"
    print("✓ Input and output shapes match")
    
    # Test with attention mask
    attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    attn_mask = attn_mask.to(device)
    output_masked = encoder(x, attn_mask=attn_mask)
    print(f"Encoder output with mask shape: {output_masked.shape}")
    print("✓ Masked attention works")
    
    # Test attention weights from all layers
    attention_weights = encoder.get_attention_weights(x)
    print(f"\nRetrieved attention weights from {len(attention_weights)} layers")
    for i, attn_weights in enumerate(attention_weights):
        if attn_weights.dim() == 4:
            print(f"  Layer {i+1}: shape {attn_weights.shape} (per head)")
        else:
            print(f"  Layer {i+1}: shape {attn_weights.shape} (averaged)")
    print("✓ Attention weights can be retrieved from all layers")
    
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
            print("   Consider further reducing parameters for encoder test.")
        else:
            print("✓ GPU memory usage is within safe limits for RTX 4050 6GB")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nEncoder total parameters: {total_params:,}")
    print(f"Encoder trainable parameters: {trainable_params:,}")
    
    # Calculate memory estimate for parameters (assuming float32)
    param_memory_mb = (total_params * 4) / (1024 ** 2)  # 4 bytes per float32
    print(f"Estimated parameter memory: {param_memory_mb:.2f} MB")
    
    # Compare with single block
    single_block = MiniTransformerBlock(d_model, n_heads, dim_feedforward)
    block_params = sum(p.numel() for p in single_block.parameters())
    print(f"\nSingle block parameters: {block_params:,}")
    print(f"Encoder is {num_layers}x larger than single block")
    
    print("\n✅ All encoder tests passed! MiniTransformerEncoder is working correctly.")
    print("\n📊 Summary of encoder parameters for RTX 4050 6GB:")
    print(f"   • Batch size: {batch_size} (reduced to 1 for 3-layer safety)")
    print(f"   • Sequence length: {seq_len} (slightly reduced)")
    print(f"   • Model dimension: {d_model}")
    print(f"   • Attention heads: {n_heads}")
    print(f"   • Number of layers: {num_layers}")
    print(f"   • Feedforward dim: {dim_feedforward}")
    
    return encoder, x, output, attention_weights


if __name__ == "__main__":
    # Run tests when the file is executed directly
    print("Starting Mini Transformer Tests...")
    print("="*60)
    
    # Test single block
    block, x_block, output_block = test_mini_transformer_block()
    
    # Test encoder with 3 stacked blocks
    encoder, x_encoder, output_encoder, attention_weights = test_mini_transformer_encoder()

    # Test full encoder-decoder forward
    test_full_transformer_forward()
    
    print("\n" + "="*60)
    print("🎉 All tests completed successfully!")
    print("Mini Transformer Block and Encoder are ready for use.")
    print("="*60)