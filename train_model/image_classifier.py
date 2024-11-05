import pandas as pd
import os
import time
import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(data_dir, batch_size):
    """Load dataset from the specified directory and return DataLoader."""
    classes = os.listdir(data_dir)
    print("Classes:", classes)

    transformations = transforms.Compose([
        transforms.Resize((512, 384)),  # Resize images to the target size
        transforms.ToTensor(),           # Convert images to tensors
    ])

    # Use ImageFolder to load all images in subdirectories
    dataset = ImageFolder(data_dir, transform=transformations)
    class_counts = {cls: 0 for cls in classes}

    for _, label in dataset.imgs:
        class_counts[classes[label]] += 1
    for cls, count in class_counts.items():
        print(f"Class: {cls}, Number of Photos: {count}")

    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader, classes

class SimpleCNN(nn.Module):
    """Define a simple CNN model."""
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 64 * 48, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 64 * 48)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def model_exists(model_save_dir):
    """Check if the model and class names exist in the specified directory."""
    model_save_path = os.path.join(model_save_dir, 'model.pt')
    class_save_path = os.path.join(model_save_dir, 'classes.json')

    model_exists = os.path.isfile(model_save_path)
    class_exists = os.path.isfile(class_save_path)

    return model_exists and class_exists  # Return True if both files exist

def train_model(model_save_dir, data_dir, train_loader, num_epochs=1):
    """Train the model on the dataset or load an existing model if available."""
    model_exists_flag = model_exists(model_save_dir)

    if model_exists_flag:
        # Load the existing model and classes
        model, classes = load_model(model_save_dir)
        print("Loaded existing model for training.")
    else:
        # If model does not exist, initialize a new model
        classes = os.listdir(data_dir)  # Assuming data_dir is accessible here
        model = SimpleCNN(num_classes=len(classes))
        print("Initialized a new model for training.")

    model.to(device)  # Move the model to the specified device

    # Calculate class weights
    class_counts = [0] * len(classes)
    for _, label in train_loader.dataset.imgs:  # Access the dataset directly
        class_counts[label] += 1

    # Calculate weights inversely proportional to class frequencies
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(classes) * count) for count in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)  # Convert to tensor and move to device

    # Update criterion to use weighted loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # For multi-class classification with weights
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()  # Start timing

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions * 100  # Calculate accuracy as a percentage
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    save_model(model, model_save_dir, classes)

def save_model(model, model_save_dir, classes):
    """Save the model and class names to the specified directory."""
    # Ensure the save directory exists
    os.makedirs(model_save_dir, exist_ok=True)

    # Define file paths
    model_save_path = os.path.join(model_save_dir, 'model.pt')
    class_save_path = os.path.join(model_save_dir, 'classes.json')

    # Save the entire model (including architecture) as a Python object
    torch.save(model, model_save_path)

    # Save class names to a JSON file
    with open(class_save_path, 'w') as f:
        json.dump(classes, f)

    print(f"Model and class names saved to {model_save_path} and {class_save_path}")

def load_model(model_save_dir):
    """Load the model and class names from the given directory."""
    # Define file names
    model_save_path = os.path.join(model_save_dir, 'model.pt')
    class_save_path = os.path.join(model_save_dir, 'classes.json')

    # Load class names from JSON file
    with open(class_save_path, 'r') as f:
        classes = json.load(f)

    # Load the entire model
    model = torch.load(model_save_path)  # Load the model directly
    model.to(device)  # Move the model to the specified device
    model.eval()  # Set the model to evaluation mode

    return model, classes  # Return model and class names

def preprocess_image(image_path, target_size=(512, 384)):
    """Preprocess the image for prediction."""
    try:
        img = Image.open(image_path).resize(target_size, Image.LANCZOS)  # Directly resize
        img_tensor = transforms.ToTensor()(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension
        return img_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict(model, image_tensor, classes, k=3):
    """Predict the top-k class probabilities for the given image tensor."""
    with torch.no_grad():  # Disable gradient calculation
        model.eval()  # Set the model to evaluation mode

        # Move the image tensor to the same device as the model
        image_tensor = image_tensor.to(device)

        outputs = model(image_tensor)  # Get the model predictions

        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0] * 100  # Get probabilities in percentages

        # Get the top-k indices and probabilities
        top_k_indices = probabilities.argsort()[-k:][::-1]  # Indices of top-k probabilities
        top_k_probs = probabilities[top_k_indices]  # Top-k probabilities

        # Get the top-k class names
        top_k_classes = [classes[i] for i in top_k_indices]

        # Return the top-k classes and their probabilities
        return list(zip(top_k_classes, top_k_probs))