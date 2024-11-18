import os
import torch
import torchvision
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights


torch.manual_seed(42)
transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])



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
        self.network = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.network.fc.in_features
        print(f'num_ftrs: {num_ftrs}, num_classes: {num_classes}')
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

def save_metadata(epoch, classes, path='saved_models', filename='metadata.txt'):
    os.makedirs(path, exist_ok=True)
    metadata_path = os.path.join(path, filename)
    with open(metadata_path, 'w') as f:
        f.write(f"Epoch: {epoch}\n")
        f.write("Classes:\n")
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"Metadata saved to {metadata_path}")

def load_metadata(path='saved_models', filename='metadata.txt'):
    metadata_path = os.path.join(path, filename)
    if not os.path.exists(metadata_path):
        print("Metadata file not found.")
        return None, None
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
        epoch = int(lines[0].split(":")[1].strip())
        classes = [line.strip() for line in lines[2:]]
    print(f"Metadata loaded from {metadata_path}")
    return epoch, classes


# Model saving and loading functions
def save_model(model, path='saved_models', filename='model.pth', epoch=0, classes=None):
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, filename)
    torch.save(model.state_dict(), model_path)
    metadata_path = os.path.join(path, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write(f"epoch: {epoch}\n")
        f.write(f"classes: {','.join(classes)}\n")
    print(f"Model and metadata saved to {path}")

def check_if_model_exists(path):
    model_path = os.path.join(path, 'model.pth')
    metadata_path = os.path.join(path, 'metadata.txt')
    return os.path.exists(model_path) and os.path.exists(metadata_path)

def load_model(path='saved_models', filename='model.pth'):
    if not check_if_model_exists(path):
        print("Model doesnt exist cant load")
    
    model_path = os.path.join(path, filename)
    metadata_path = os.path.join(path, "metadata.txt")

    if os.path.exists(model_path) and os.path.exists(metadata_path):
        

        # Load metadata
        with open(metadata_path, "r") as f:
            lines = f.readlines()
            epoch = int(lines[0].split(":")[1].strip())
            classes = lines[1].split(":")[1].strip().split(',')

        print(f"Metadata loaded: Epoch {epoch}, Classes: {classes}")
        model = ResNet(num_classes=len(classes)) 
        model.load_state_dict(torch.load(model_path, map_location=get_default_device()))
        print(f"Model loaded from {model_path}")

        return model, epoch, classes
    else:
        print("Pretrained model not found.")
        return None, 0, []

def initialize_model(num_classes, path='saved_models', filename='model.pth'):
    model = ResNet(num_classes)
    epoch, classes = 0, []
    print("Initializing a new model.")
    save_model(model, path, filename, epoch, classes)
    return model, epoch, classes

# Training and evaluation functions
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(args, model, model_dir, classes, train_loader, val_loader, opt_func=torch.optim.SGD, current_epochs=0):
    history = []
    optimizer = opt_func(model.parameters(), args['learning_rate'])
    for epoch in range(args['num_epochs']):
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
        print(f"classes: {classes}")
        save_model(model=model, path=model_dir, epoch=(current_epochs + epoch + 1), classes=classes)
        history.append(result)
    return history

def create_args(num_epochs=1, batch_size=32, learning_rate=1e-5, save_every=500, max_length=256, temperature=1.0, top_k=5, top_p=0.9, repetition_penalty=1.0):
    """
    Returns a dictionary of training and evaluation arguments.
    
    Args:
        num_epochs (int, optional): Number of epochs. Defaults to 1.
        batch_size (int, optional): Batch size. Defaults to 32.
        learning_rate (float, optional): Learning rate. Defaults to 1e-5.
        save_every (int, optional): Save model every X steps. Defaults to 500.
        max_length (int, optional): Maximum sequence length. Defaults to 256.
        temperature (float, optional): Sampling temperature for predictions. Defaults to 1.0.
        top_k (int, optional): Number of top predictions to consider. Defaults to 5.
        top_p (float, optional): Cumulative probability for nucleus sampling. Defaults to 0.9.
        repetition_penalty (float, optional): Penalty for repeated phrases. Defaults to 1.0.
        data_dir (str, optional): Path to the dataset directory. Defaults to 'data/garbage_classification'.
        model_dir (str, optional): Directory to save the model. Defaults to 'saved_models'.
        test_image_path (str, optional): Path to test image for prediction. Defaults to None.

    Returns:
        dict: Dictionary containing training and evaluation arguments.
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


def train(data_dir=None, model_dir=None, args=create_args(), file_name='model.pth'):
    device = get_default_device()

    classes = os.listdir(data_dir)
    dataset = ImageFolder(data_dir, transform=transformations)
    train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
    train_dl = DataLoader(train_ds, args["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, args["batch_size"] * 2, num_workers=0, pin_memory=True)
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    model, epoch = None, 0

    if not check_if_model_exists(model_dir):
        print("Model doesnt exist. Creating a new model!")
        model, _, _ = initialize_model(len(classes), path=model_dir)
    else:
        print("Found Model loading model")
        model, epoch, classes = load_model(model_dir, filename=file_name)
    model = to_device(model, device)
    opt_func = torch.optim.Adam
    history = fit(args, model, model_dir, classes, train_dl, val_dl, opt_func)
    return history

# Top-k prediction function for a single image
def predict_images(img_path, model, classes, args):
    device = get_default_device()
    
    model = to_device(model, device)
    
    if not classes:
        print("Class labels not found in metadata.")
        return
    
    image = Image.open(img_path).convert('RGB')
    image_transformed = transformations(image)
    image_transformed = to_device(image_transformed.unsqueeze(0), device)

    model.eval()
    with torch.no_grad():
        output = model(image_transformed)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_k_preds = torch.topk(probabilities, args['top_k'], dim=1)
    
    top_k_labels = [classes[idx] for idx in top_k_preds[0].tolist()]
    top_k_probs = top_probs[0].tolist()

    return top_k_labels, top_k_probs

if __name__ == '__main__':
    data_dir = 'data/garbage_classification'
    model_dir = 'saved_models'

    args = create_args(
        num_epochs=2, 
        batch_size=32, 
        learning_rate=1e-5, 
        save_every=500, 
        max_length=256, 
        temperature=1.0, 
        top_k=5, 
        top_p=0.9, 
        repetition_penalty=1.0
        )
    
    train(data_dir=data_dir, model_dir=model_dir, args=args)
    #12 epochs (sweetspot for now)
    model, epoch, classes = load_model(model_dir)
    # Test Top-k prediction
    test_image_path = r'C:\Users\minec\Downloads\test\metal.jpg'
    predict_images(test_image_path, model, classes, args)
