import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def train_data(curri, anti_curri):
    """
    Returns DataLoader objects for training datasets based on a curriculum learning, anti-curriculum learning or random strategy.

    Parameters:
    - curri (bool): If True, applies a curriculum learning strategy by introducing simpler classes first.
    - anti_curri (bool): If True, applies an anti-curriculum learning strategy by starting with harder classes.
    
    Returns:
    - If curri or anti_curri is True: returns three DataLoader objects (level1_loader, level2_loader, level3_loader) 
      corresponding to increasing or decreasing difficulty levels.
    - If both curri and anti_curri are False: returns a single DataLoader with the full training set.
    """

    # Define image transformations: resize images and convert them to tensors
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # function to extract a subset of the dataset based on class names
    def get_subset(folder, class_names):
        full_dataset = ImageFolder(folder, transform=transform)
        indices = [i for i, (_, label) in enumerate(full_dataset) if full_dataset.classes[label] in class_names]
        subset = torch.utils.data.Subset(full_dataset, indices)
        return subset

    # Case 1: Anti-curriculum learning (start with hard examples)
    if anti_curri and not curri:
        level_1 = get_subset('dataset/train', ['kite', 'rhombus','trapezoid'])
        level_2 = get_subset('dataset/train', ['kite', 'rhombus','trapezoid','rectangle', 'parallelogram'])
        level_3 = get_subset('dataset/train', ['kite', 'rhombus','trapezoid','rectangle', 'parallelogram','square', 'triangle', 'circle'])  
        level1_loader = DataLoader(level_1, batch_size=32, shuffle=True)
        level2_loader = DataLoader(level_2, batch_size=32, shuffle=True)
        level3_loader = DataLoader(level_3, batch_size=32, shuffle=True)
        return level1_loader, level2_loader, level3_loader 
    
    # Case 2: Curriculum learning (start with easy examples)
    elif curri and not anti_curri:
        level_1 = get_subset('dataset/train', ['square', 'triangle', 'circle'])
        level_2 = get_subset('dataset/train', ['square', 'triangle', 'circle','rectangle', 'parallelogram'])
        level_3 = get_subset('dataset/train', ['kite', 'rhombus','trapezoid','rectangle', 'parallelogram','square', 'triangle', 'circle']) 
        level1_loader = DataLoader(level_1, batch_size=32, shuffle=True)
        level2_loader = DataLoader(level_2, batch_size=32, shuffle=True)
        level3_loader = DataLoader(level_3, batch_size=32, shuffle=True)
        return level1_loader, level2_loader, level3_loader 

    # Case 3: No curriculum strategy, return full dataset
    else:
        level = get_subset('dataset/train', ['kite', 'rhombus','trapezoid','rectangle', 'parallelogram','square', 'triangle', 'circle']) 
        all_loader = DataLoader(level, batch_size=32, shuffle=True)
        return all_loader


def test_data():
    """
    Loads and returns the test dataset using a PyTorch DataLoader.

    The function applies preprocessing transformations to the test images, including resizing and converting to tensors.
    It returns a DataLoader that batches the data and shuffles it for evaluation.

    Returns:
    - test_loader (DataLoader): A PyTorch DataLoader containing the test dataset with batch size 32.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    test_dataset = ImageFolder('dataset/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32 , shuffle = True)
    return test_loader 


class ShapeCNN_1(nn.Module):
    """
    A CNN for image classification.

    Architecture:
    - Two convolutional layers with ReLU activation and max pooling.
    - One fully connected hidden layer.
    - One output layer for classification.

    Parameters:
    - num_classes (int): The number of output classes for classification.

    Input shape: (batch_size, 3, 64, 64)
    Output shape: (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()

        # First convolutional layer:
        # Input channels = 3 (RGB image)
        # Output channels = 32
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2) # Reduces feature map to 2x2

        # Second convolutional layer:
        # Input channels = 32
        # Output channels = 64
        self.conv2 = nn.Conv2d(32, 64, 3)

        # Fully connected layer:
        # Input features = 64 channels * 14 * 14 
        # Output features = 128
        self.fc1 = nn.Linear(64 * 14 * 14, 128)

        # Output layer:
        # Input features = 128
        # Output features = num_classes 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply first conv layer, ReLU, then pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second conv layer, ReLU, then pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 14 * 14)
        # Apply first fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ShapeCNN_2 (nn.Module):
    """
    A lightweight Convolutional Neural Network (CNN) for image classification.

    Architecture:
    - A single convolutional layer with ReLU activation.
    - A global average pooling layer to compress spatial information.
    - A fully connected linear layer that outputs class logits.

    Parameters:
    - num_classes (int): The number of output classes for classification.

    Input shape: (batch_size, 3, H, W)
    Output shape: (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # One convolutional layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduces feature map to 1x1
        self.fc = nn.Linear(16, num_classes)  # Fully connected layer

    def forward(self, x):
        x = F.relu(self.conv(x))  # Apply convolution + ReLU
        x = self.pool(x)          # Apply global average pooling
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)         


class ShapeCNN_3(nn.Module):
    """
    A minimalist CNN for image classification using global average pooling.

    Architecture:
    - A single convolutional layer to directly map the input to the number of classes.
    - A global average pooling layer to reduce each feature map to a single value.
    - The output is a flat vector of class scores.

    Parameters:
    - num_classes (int): The number of output classes for classification.

    Input shape: (batch_size, 3, H, W)
    Output shape: (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

    def forward(self, x):
        x = self.conv(x)         
        x = self.pool(x)          # Reduce to 1x1
        x = x.view(x.size(0), -1) # Flatten
        return x                 


def train(model, dataloader, criterion, optimizer):
    """
    Trains the given model for one epoch using the provided dataloader, loss function, and optimizer.

    Args:
        model : The neural network model to train.
        dataloader : DataLoader providing batches of training data.
        criterion : Loss function to compute the error (e.g., nn.CrossEntropyLoss).
        optimizer : Optimizer used to update model parameters.

    Returns:
        tuple:
            - batch_losses (list of float): The loss value for each batch during the epoch.
            - avg_loss (float): The average loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0
    batch_losses = [] # Stores loss for each batch for monitoring
    
    for inputs, labels in dataloader:
        inputs, labels = inputs, labels

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels) # Compute the loss between predictions and true labels
        loss.backward()
        optimizer.step() # Update model parameters using gradients

        total_loss += loss.item()
        batch_losses.append(loss.item())

    avg_loss = total_loss / len(dataloader)
    return batch_losses, avg_loss


def evaluate(model, loader):
    """
    Evaluates the accuracy of a trained model on a given dataset.

    Args:
        model (nn.Module): The trained neural network model to evaluate.
        loader (DataLoader): DataLoader providing batches of data to evaluate the model on.

    Returns:
        float: The accuracy of the model on the dataset (correct predictions / total samples).
    """ 

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images, labels
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # Get predicted class index
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total



def subset_to_tensor(loader):
    all_images, all_labels = [], []
    for images, labels in loader:
        all_images.append(images)
        all_labels.append(labels)
    return torch.cat(all_images), torch.cat(all_labels)

def normalize_data(X):
    mean = X.mean(dim=0)
    std = X.std(dim=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def train_logistic (model , data_loader):
    X, y = subset_to_tensor(data_loader)
    X = X.view (X.size(0), -1)
    X = normalize_data(X)
    mask = ~torch.isnan(X).any(dim=1)
    X = X[mask]
    model.fit(X.numpy(), y.numpy())
    

def evaluate_logistic(model, loader):
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.view(images.size(0), -1)  
        outputs = model.predict(images.numpy())   
        total += labels.size(0)
        correct += (outputs == labels.numpy()).sum().item()
    return correct / total
