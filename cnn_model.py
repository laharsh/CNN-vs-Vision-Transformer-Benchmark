import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Convolutional Neural Network for Image Classification
    
    This CNN architecture demonstrates the key components:
    1. Convolutional layers for feature extraction
    2. Pooling layers for dimension reduction
    3. Fully connected layers for classification
    
    Architecture:
    Input (32x32x3) -> Conv1 -> ReLU -> MaxPool -> Conv2 -> ReLU -> MaxPool -> 
    Conv3 -> ReLU -> MaxPool -> Flatten -> FC1 -> ReLU -> Dropout -> FC2 -> Output
    """
    
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # First Convolutional Block
        # Input: 32x32x3 (RGB image)
        # Output: 32x32x32 (32 feature maps)
        self.conv1 = nn.Conv2d(
            in_channels=3,      # RGB channels
            out_channels=32,    # Number of feature maps to learn
            kernel_size=3,      # 3x3 filter size
            padding=1           # Keep spatial dimensions same
        )
        
        # Second Convolutional Block
        # Input: 32x32x32
        # Output: 16x16x64 (after pooling)
        self.conv2 = nn.Conv2d(
            in_channels=32,     # Input from previous layer
            out_channels=64,    # More feature maps for complex patterns
            kernel_size=3,
            padding=1
        )
        
        # Third Convolutional Block
        # Input: 16x16x64
        # Output: 8x8x128 (after pooling)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,   # Even more feature maps
            kernel_size=3,
            padding=1
        )
        
        # Max Pooling layers
        # Reduce spatial dimensions by 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # We'll calculate the flattened size dynamically in forward pass
        # to avoid hardcoding dimensions
        self.fc1 = None  # Will be set after first forward pass
        self.fc2 = nn.Linear(512, num_classes)   # Final classification layer
        
        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.5)
        
        # Flag to track if fc1 has been initialized
        self.fc1_initialized = False
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        
        # First conv block: 32x32x3 -> 32x32x32
        x = self.pool(F.relu(self.conv1(x)))
        # ReLU activation: max(0, x) - introduces non-linearity
        # MaxPool: takes maximum value in 2x2 windows
        
        # Second conv block: 32x32x32 -> 16x16x64
        x = self.pool(F.relu(self.conv2(x)))
        
        # Third conv block: 16x16x64 -> 8x8x128
        x = self.pool(F.relu(self.conv3(x)))
        
        # Get the current batch size and calculate flattened size
        batch_size = x.size(0)
        flattened_size = x.size(1) * x.size(2) * x.size(3)
        
        # Initialize fc1 layer on first forward pass
        if not self.fc1_initialized:
            self.fc1 = nn.Linear(flattened_size, 512).to(x.device)
            self.fc1_initialized = True
            print(f"Initialized fc1 layer with input size: {flattened_size}")
        
        # Flatten the feature maps for fully connected layers
        x = x.view(batch_size, -1)
        
        # First fully connected layer
        x = F.relu(self.fc1(x))
        
        # Dropout for regularization
        x = self.dropout(x)
        
        # Final classification layer
        x = self.fc2(x)
        
        return x

def create_cnn_model(num_classes=10, device='cpu'):
    """
    Create and initialize a CNN model
    
    Args:
        num_classes: Number of output classes
        device: Device to place the model on ('cpu' or 'cuda')
        
    Returns:
        Initialized CNN model
    """
    model = CNN(num_classes=num_classes)
    model = model.to(device)
    
    # Initialize weights for better training
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Xavier initialization for convolutional layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear) and m is not None:
            # Xavier initialization for linear layers (skip None fc1)
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
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
    # Create a sample model
    model = create_cnn_model()
    
    # Create a dummy input (batch_size=1, channels=3, height=32, width=32)
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Forward pass (this will initialize fc1)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
