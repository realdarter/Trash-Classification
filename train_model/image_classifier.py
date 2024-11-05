import os
import time
import json
import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from PIL import Image

scaler = torch.cuda.amp.GradScaler()  # No need to pass 'cuda' as a string


# Function to create training arguments
def create_args(num_epochs=1, batch_size=32, learning_rate=5e-5, save_every=3, 
                max_length=512, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2):
    """
    Returns a dictionary of training arguments.
    Args:
        num_epochs (int, optional): Number of epochs. Defaults to 1.
        batch_size (int, optional): Batch size. Defaults to 32.
        learning_rate (float, optional): Learning rate. Defaults to 5e-5.
        save_every (int, optional): Save model every X steps. Defaults to 500.
        max_length (int, optional): Maximum length of generated sequences. Defaults to 512.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        top_k (int, optional): Top-K sampling. Defaults to 50.
        top_p (float, optional): Top-P (nucleus) sampling. Defaults to 0.95.
        repetition_penalty (float, optional): Repetition penalty. Defaults to 1.2.
    
    Returns:
        dict: Dictionary containing training arguments.
    """
    return {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "save_every": save_every,
        "max_length": max_length,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty
    }

# Hyperparameters
args = create_args(num_epochs=20, batch_size=32, learning_rate=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(data_dir, batch_size):
    """Load dataset from the specified directory and return DataLoader."""
    classes = os.listdir(data_dir)
    print("Classes:", classes)

    transformations = transforms.Compose([
    transforms.Resize((512, 384)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Randomly rotate images
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly change brightness and color
    transforms.RandomResizedCrop(size=(512, 384), scale=(0.8, 1.0)),  # Randomly crop images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])



    # Use ImageFolder to load all images in subdirectories
    dataset = ImageFolder(data_dir, transform=transformations)
    class_counts = count_classes(dataset, classes)

    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, classes

def count_classes(dataset, classes):
    """Count the number of instances per class in the dataset."""
    class_counts = {cls: 0 for cls in classes}
    for _, label in dataset.imgs:
        class_counts[classes[label]] += 1
    for cls, count in class_counts.items():
        print(f"Class: {cls}, Number of Photos: {count}")
    return class_counts

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
    return os.path.isfile(model_save_path) and os.path.isfile(class_save_path)

def train_model(model_save_dir, data_dir, train_loader, args):
    model_exists_flag = model_exists(model_save_dir)

    if model_exists_flag:
        model, classes = load_model(model_save_dir)
    else:
        classes = os.listdir(data_dir)
        model = SimpleCNN(num_classes=len(classes))

    model.to(device)

    class_counts = count_classes(train_loader.dataset, classes)
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([total_samples / (len(classes) * count) if count > 0 else 0 for count in class_counts.values()], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler()

    best_accuracy = 0.0
    patience = 5  # Early stopping patience
    patience_counter = 0

    for epoch in range(args['num_epochs']):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions * 100

        print(f"Epoch [{epoch + 1}/{args['num_epochs']}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Early stopping check
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            save_model(model, model_save_dir, classes)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping")
            break

        scheduler.step()  # Update learning rate

    print("Training complete.")


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
    """Predict the top-k class probabilities for the given image."""
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        image_tensor = image_tensor.to(device)  # Move the tensor to the specified device
        outputs = model(image_tensor)
        probabilities = nn.functional.softmax(outputs, dim=1).cpu().numpy()  # Convert to probabilities
        top_k_indices = probabilities.argsort()[0][-k:][::-1]  # Get indices of top-k predictions
        top_k_probs = probabilities[0][top_k_indices]  # Get top-k probabilities
        return [(classes[idx], prob) for idx, prob in zip(top_k_indices, top_k_probs)]

def test_model(model_save_dir, data_dir, args):
    """Test the model on the test dataset."""
    test_loader, classes = load_data(data_dir, args['batch_size'])

    # Load the trained model
    model, _ = load_model(model_save_dir)  # Load the existing model

    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    criterion = nn.CrossEntropyLoss()  # Assume you want to use the same loss function

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions * 100  # Calculate accuracy as a percentage
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
