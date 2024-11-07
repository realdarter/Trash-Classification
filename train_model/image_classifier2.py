import os
import torch
import torchvision
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt

data_dir = 'data/garbage_classification'
classes = os.listdir(data_dir)

# Define transformations for the images
transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Load the dataset
dataset = ImageFolder(data_dir, transform=transformations)


# Helper function to show an image sample
def show_sample(img, label, dataset):
    print("Label:", dataset.classes[label], "(Class No: "+ str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))


# Define accuracy functions
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def top_k_accuracy(outputs, labels, k=5):
    _, top_k_preds = torch.topk(outputs, k, dim=1)
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))
    return torch.tensor(correct.any(dim=1).float().mean().item())


# Base model class
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
        acc_top1 = accuracy(out, labels)
        acc_top5 = top_k_accuracy(out, labels, k=5)
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


# Define the ResNet model
class ResNet(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, xb):
        return torch.softmax(self.network(xb), dim=1)


# Device management functions
def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


# Model saving and loading functions
def save_model(model, path='saved_models', filename='model.pth'):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, filename))
    print(f"Model saved to {os.path.join(path, filename)}")

def load_model(model, path='saved_models', filename='model.pth'):
    model.load_state_dict(torch.load(os.path.join(path, filename), map_location=get_default_device()))
    print(f"Model loaded from {os.path.join(path, filename)}")
    return model


# Training and evaluation functions
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# Top-k prediction function for a single image
def predict_top_k(img_path, model, transformations, k=5, device=None):
    if device is None:
        device = get_default_device()
    
    image = Image.open(img_path).convert('RGB')
    image_transformed = transformations(image)
    image_transformed = to_device(image_transformed.unsqueeze(0), device)

    model.eval()
    with torch.no_grad():
        output = model(image_transformed)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_k_preds = torch.topk(probabilities, k, dim=1)
    
    top_k_labels = [dataset.classes[idx] for idx in top_k_preds[0].tolist()]
    top_k_probs = top_probs[0].tolist()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Top-{k} Predictions:\n" + "\n".join([f"{label}: {prob*100:.2f}%" for label, prob in zip(top_k_labels, top_k_probs)]))
    plt.axis('off')
    plt.show()
    return top_k_labels, top_k_probs


if __name__ == '__main__':
    random_seed = 42
    torch.manual_seed(random_seed)
    train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
    batch_size = 32
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)
    
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    # Initialize model, move to device, and start training
    model = ResNet(num_classes=len(classes))
    model = to_device(model, device)
    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = 1.0e-5
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    # Save the model
    save_model(model)

    # Load the model (for testing)
    model = load_model(model)
    
    # Test Top-k prediction
    test_image_path = 'test_image.jpg'  # Replace with your image path
    predict_top_k(test_image_path, model, transformations, k=5, device=device)
