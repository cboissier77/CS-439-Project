import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


def new_train_data_curriculum():
    """ Function to create a curriculum learning dataset for training.
    This function organizes the dataset into three levels of complexity based on the shapes present in the dataset.
    Returns:
        level1_loader: DataLoader for the first level of complexity (kite, rectangles).
        level2_loader: DataLoader for the second level of complexity (kite, rhombus, square, rectangle).
        level3_loader: DataLoader for the third level of complexity (kite, rhombus, trapezoid, rectangle, parallelogram, square, triangle, circle).
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
    
    level_1 = get_subset('dataset/train', ['rhombus', 'kite','circle'])
    level_2 = get_subset('dataset/train', ['rhombus', 'kite','circle', 'square','triangle'])
    level_3 = get_subset('dataset/train', ['kite', 'rhombus','trapezoid','rectangle', 'parallelogram','square', 'triangle', 'circle'])  
    level1_loader = DataLoader(level_1, batch_size=32, shuffle=True)
    level2_loader = DataLoader(level_2, batch_size=32, shuffle=True)
    level3_loader = DataLoader(level_3, batch_size=32, shuffle=True)
    return level1_loader, level2_loader, level3_loader 
    


def standard_curriculum ():
    """ Function to create a standard curriculum learning dataset for training.
    This function organizes the dataset into one level of complexity based on the shapes present in the dataset.
    Returns:
        level_loader: DataLoader for the dataset containing all shapes.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    def get_subset(folder, class_names):
        full_dataset = ImageFolder(folder, transform=transform)
        indices = [i for i, (_, label) in enumerate(full_dataset) if full_dataset.classes[label] in class_names]
        subset = torch.utils.data.Subset(full_dataset, indices)
        return subset
    
    level = get_subset('dataset/train', ['square', 'triangle', 'circle','rectangle', 'parallelogram' , 'kite', 'rhombus','trapezoid'])
    level_loader = DataLoader(level, batch_size=32, shuffle=False)
    return level_loader 