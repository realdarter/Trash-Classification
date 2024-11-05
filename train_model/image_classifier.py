import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from PIL import Image

data_dir = 'data/garbage_classification'
classes = os.listdir(data_dir)
print(classes)

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
dataset = ImageFolder(data_dir, transform=transformations)

def show_sample(img, label):
    print("Label:", dataset.classes[label], "(Class No: "+ str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))

random_seed = 42
torch.manual_seed(random_seed)

train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
batch_size = 32
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)

from torchvision.utils import make_grid

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def top_k_accuracy(outputs, labels, k=5):
    # Get the top k predictions
    _, top_k_preds = torch.topk(outputs, k, dim=1)
    # Check if the true label is in the top k predictions
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))
    # Calculate the accuracy as the mean of correct predictions
    return torch.tensor(correct.any(dim=1).float().mean().item())


class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc_top1 = accuracy(out, labels)  # Top-1 accuracy
        acc_top5 = top_k_accuracy(out, labels, k=5)  # Top-5 accuracy
        return {'val_loss': loss.detach(), 'val_acc_top1': acc_top1, 'val_acc_top5': acc_top5}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc_top1 = [x['val_acc_top1'] for x in outputs]
        epoch_acc_top1 = torch.stack(batch_acc_top1).mean()
        batch_acc_top5 = [x['val_acc_top5'] for x in outputs]
        epoch_acc_top5 = torch.stack(batch_acc_top5).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc_top1': epoch_acc_top1.item(), 'val_acc_top5': epoch_acc_top5.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc_top1: {:.4f}, val_acc_top5: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc_top1'], result['val_acc_top5']))

        
class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        
        # Use a pretrained ResNet50 model
        self.network = models.resnet50(pretrained=True)  # Set pretrained=True to load the pretrained weights
        
        # Replace the last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(classes))
    
    def forward(self, xb):
        return torch.softmax(self.network(xb), dim=1)

model = ResNet()

def get_default_device():
    
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def predict_top_k(img, model, k=5):
    xb = to_device(img.unsqueeze(0), device)  # Convert to a batch of 1
    yb = model(xb)  # Get model output
    _, top_k_preds = torch.topk(yb, k, dim=1)  # Get top k predictions
    top_k_labels = [dataset.classes[idx] for idx in top_k_preds[0].tolist()]  # Get class labels
    return top_k_labels


device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
model = to_device(model, device)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

if __name__ == '__main__':
    # Initialize and move the model to device
    model = to_device(ResNet(), device)
    
    # Evaluate the model before training
    evaluate(model, val_dl)
    
    # Training parameters
    num_epochs = 60
    opt_func = torch.optim.Adam
    lr = 1.0e-5 #5.5e-5

    # Train the model
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    
    # Testing Top-k Prediction for a Single Image
    k = 5  # Set your desired Top-k value

    # Load and transform the image using the existing transformations
    image_path = r'C:\Users\minec\Downloads\test\metal.jpg'  # Replace with your image path
    image = Image.open(image_path).convert('RGB')  # Load image
    image_transformed = transformations(image)  # Apply the existing transformations

    # Move the image to the appropriate device and make a prediction
    image_transformed = to_device(image_transformed.unsqueeze(0), device)  # Add batch dimension

    # Set the model to evaluation mode and make a prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        output = model(image_transformed)  # Get the model output
        probabilities = torch.softmax(output, dim=1)  # Convert to probabilities using softmax
        top_probs, top_k_preds = torch.topk(probabilities, k, dim=1)  # Get top-k probabilities and predictions
    
    # Retrieve the class labels and associated probabilities for Top-k predictions
    top_k_labels = [dataset.classes[idx] for idx in top_k_preds[0].tolist()]
    top_k_probs = top_probs[0].tolist()
    
    # Display the image and the Top-k predictions with probabilities
    plt.figure(figsize=(6, 6))
    plt.imshow(image)  # Show the original image
    plt.title(f"Top-{k} Predictions:\n" + "\n".join([f"{label}: {prob*100:.2f}%" for label, prob in zip(top_k_labels, top_k_probs)]))
    plt.axis('off')  # Hide axes
    plt.show()
