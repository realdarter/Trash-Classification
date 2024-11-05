import os
import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18  # Example model

def create_args(num_epochs=1, batch_size=32, learning_rate=5e-5, 
                image_size=(224, 224), validation_split=0.2):
    return {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "image_size": image_size,
        "validation_split": validation_split
    }

def train_model(data_dir, model_dir, args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize(args["image_size"]),
        transforms.ToTensor()
    ])

    # Load the dataset
    dataset = ImageFolder(root=data_dir, transform=transform)

    # Print classes and number of images in each class
    class_counts = {class_name: 0 for class_name in dataset.classes}
    
    for _, label in dataset:
        class_name = dataset.classes[label]
        class_counts[class_name] += 1

    print("Classes and number of images in each class:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")

    num_samples = len(dataset)
    num_val_samples = int(num_samples * args["validation_split"])
    num_train_samples = num_samples - num_val_samples
    
    train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])
    
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)

    # Initialize model
    model = resnet18(pretrained=False, num_classes=1).to(device)  # Set num_classes to 1

    # Check if model exists and load it
    model_path = os.path.join(model_dir, 'model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model from disk.")
    else:
        print("No existing model found. Starting training from scratch.")

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Use binary cross-entropy loss for a single class
    optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"])

    # Training loop
    for epoch in range(args["num_epochs"]):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Inside your training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()  # Convert labels to float for BCE

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images).squeeze()  # Squeeze to remove extra dimensions

            # Debugging output
            print(f"Outputs: {outputs}")  # Check raw logits
            print(f"Labels: {labels}")      # Check labels

            loss = criterion(outputs, labels)

            # Check loss value
            print(f"Loss: {loss.item()}")  # Should be a positive value

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Calculate average loss and accuracy
        avg_loss = running_loss / len(train_loader)
        if total > 0:
            accuracy = 100 * correct / total
        else:
            accuracy = 0.0  # or some other value that makes sense in your context

        
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{args['num_epochs']}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Save the model once at the end of training
    torch.save(model.state_dict(), model_path)
    print("Training complete. Model saved.")

def load_model(model_dir):
    model = resnet18(pretrained=False, num_classes=1)  # Set num_classes to 1
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
    return model

# Example usage
if __name__ == "__main__":
    data_dir = "data/garbage_classification"  # Make sure this folder has your image
    model_dir = "checkpoint/save1"
    os.makedirs(model_dir, exist_ok=True)

    args = create_args(num_epochs=30, batch_size=32, learning_rate=1e-5)
    train_model(data_dir, model_dir, args)
