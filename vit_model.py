"""
Vision Transformer (ViT) Implementation for Image Classification

This module implements a Vision Transformer model optimized for CIFAR-10 classification.
The ViT architecture treats images as sequences of patches and applies transformer
attention mechanisms for image understanding.

Key Features:
1. Patch embedding and positional encoding
2. Multi-head self-attention
3. Layer normalization and residual connections
4. Classification head with global average pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """
    Converts image patches to embeddings
    
    Args:
        img_size: Input image size (assumed square)
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embedding: convert patches to embeddings
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, embed_dim)  # +1 for cls token
        )
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.patch_embedding(x)  # (B, embed_dim, 8, 8) for 32x32 image with 4x4 patches
        
        # Flatten spatial dimensions: (B, embed_dim, 8, 8) -> (B, embed_dim, 64)
        x = x.flatten(2)  # (B, embed_dim, 64)
        
        # Transpose: (B, embed_dim, 64) -> (B, 64, embed_dim)
        x = x.transpose(1, 2)  # (B, 64, embed_dim)
        
        # Add classification token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 65, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Apply dropout
        x = self.dropout(x)
        
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(self, embed_dim=192, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V: (B, seq_len, embed_dim) -> (B, seq_len, embed_dim*3)
        qkv = self.qkv(x)
        
        # Reshape: (B, seq_len, embed_dim*3) -> (B, seq_len, 3, num_heads, head_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        
        # Permute: (B, seq_len, 3, num_heads, head_dim) -> (3, B, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Split into Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        # Attention scores: (B, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values: (B, num_heads, seq_len, head_dim)
        attended = torch.matmul(attention_weights, v)
        
        # Concatenate heads: (B, num_heads, seq_len, head_dim) -> (B, seq_len, embed_dim)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.proj(attended)
        output = self.dropout(output)
        
        return output

class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and MLP
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embed_dim
        dropout: Dropout rate
    """
    
    def __init__(self, embed_dim=192, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer for image classification
    
    Args:
        img_size: Input image size
        patch_size: Size of each patch
        in_channels: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embed_dim
        dropout: Dropout rate
    """
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=192, depth=6, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights using Xavier uniform initialization"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Patch embedding and positional encoding
        x = self.patch_embedding(x)  # (B, 65, embed_dim)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Extract classification token (first token)
        cls_token = x[:, 0]  # (B, embed_dim)
        
        # Classification head
        logits = self.head(cls_token)  # (B, num_classes)
        
        return logits

def create_vit_model(num_classes=10, device='cpu', model_size='small'):
    """
    Create and initialize a Vision Transformer model
    
    Args:
        num_classes: Number of output classes
        device: Device to place the model on ('cpu' or 'cuda')
        model_size: Model size ('tiny', 'small', 'base')
        
    Returns:
        Initialized ViT model
    """
    
    # Model configurations
    configs = {
        'tiny': {
            'embed_dim': 192,
            'depth': 6,
            'num_heads': 8,
            'patch_size': 4
        },
        'small': {
            'embed_dim': 384,
            'depth': 8,
            'num_heads': 8,
            'patch_size': 4
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'patch_size': 4
        }
    }
    
    config = configs[model_size]
    
    model = VisionTransformer(
        img_size=32,
        patch_size=config['patch_size'],
        in_channels=3,
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=4,
        dropout=0.1
    )
    
    model = model.to(device)
    
    return model

def get_model_info(model):
    """
    Get detailed information about the model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'architecture': type(model).__name__
    }

# Example usage and testing
if __name__ == "__main__":
    # Create a sample ViT model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_vit_model(num_classes=10, device=device, model_size='small')
    
    # Create a dummy input (batch_size=1, channels=3, height=32, width=32)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get model information
    model_info = get_model_info(model)
    print(f"\nModel Information:")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    print(f"Architecture: {model_info['architecture']}")
    
    # Print model architecture
    print(f"\nModel Architecture:")
    print(model)
