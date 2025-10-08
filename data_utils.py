import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def get_data_transforms():
    """
    Get data transformation pipelines for training and testing
    
    Training transforms include data augmentation to improve generalization:
    - Random horizontal flip: Helps with orientation invariance
    - Random rotation: Helps with rotation invariance
    - Color jitter: Helps with lighting/color variations
    - Random crop: Helps with position invariance
    
    Testing transforms only normalize the data
    """
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),      # 50% chance of horizontal flip
        transforms.RandomRotation(degrees=10),       # Random rotation up to 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomCrop(32, padding=4),       # Random crop with padding
        transforms.ToTensor(),                       # Convert PIL image to tensor
        transforms.Normalize(                        # Normalize with CIFAR-10 statistics
            mean=[0.4914, 0.4822, 0.4465],         # Mean for each channel (R, G, B)
            std=[0.2023, 0.1994, 0.2010]           # Standard deviation for each channel
        )
    ])
    
    # Testing transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    return train_transform, test_transform

def load_cifar10_data(batch_size=128, num_workers=0):  # Reduced workers for Windows compatibility
    """
    Load CIFAR-10 dataset with data loaders
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
        
    Returns:
        train_loader, test_loader: Data loaders for training and testing
    """
    
    print("Loading CIFAR-10 dataset...")
    
    # Get transforms
    train_transform, test_transform = get_data_transforms()
    
    # Download and load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Shuffle training data for better training
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Only pin memory if GPU is available
        drop_last=True         # Drop last incomplete batch to ensure consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Only pin memory if GPU is available
        drop_last=True         # Drop last incomplete batch to ensure consistent batch sizes
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {CIFAR10_CLASSES}")
    
    return train_loader, test_loader

def denormalize_image(tensor):
    """
    Denormalize a normalized image tensor back to [0, 1] range
    
    Args:
        tensor: Normalized image tensor
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    # CIFAR-10 normalization values
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    # Denormalize: (x - mean) / std -> x = (x * std) + mean
    denorm = tensor * std + mean
    
    # Clamp to [0, 1] range
    denorm = torch.clamp(denorm, 0, 1)
    
    return denorm

def visualize_batch(images, labels, num_images=8, class_names=None):
    """
    Visualize a batch of images with their labels
    
    Args:
        images: Batch of image tensors
        labels: Batch of labels
        num_images: Number of images to display
        class_names: List of class names
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES
    
    # Denormalize images
    images = denormalize_image(images)
    
    # Convert to numpy for matplotlib
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        # Transpose from (C, H, W) to (H, W, C) for matplotlib
        img = np.transpose(images[i], (1, 2, 0))
        
        axes[i].imshow(img)
        axes[i].set_title(f'{class_names[labels[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_class_distribution(dataset):
    """
    Get the distribution of classes in the dataset
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        Dictionary with class counts
    """
    class_counts = {}
    
    for _, label in dataset:
        class_name = CIFAR10_CLASSES[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return class_counts

def plot_class_distribution(class_counts):
    """
    Plot the class distribution of the dataset
    
    Args:
        class_counts: Dictionary with class counts
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(count), ha='center', va='bottom')
    
    plt.title('CIFAR-10 Dataset Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load data
    train_loader, test_loader = load_cifar10_data(batch_size=8)
    
    # Get a batch and visualize
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        print(f"Labels: {labels}")
        
        # Visualize the batch
        visualize_batch(images, labels)
        break
    
    # Show class distribution
    print("\nClass distribution:")
    for i, class_name in enumerate(CIFAR10_CLASSES):
        print(f"{i}: {class_name}")
