from image_classifier import *

data_dir = 'data/garbage_classification'
batch_size = 32  # Adjust based on your GPU memory
model_save_path = 'checkpoint/save1'

# Load the data for training
train_loader, classes = load_data(data_dir, batch_size)

# Initialize and set up the model
model = SimpleCNN(num_classes=len(classes)).to(device)

# Train the model
train_model(model_save_path, data_dir, train_loader, num_epochs=20)

# Save the model and classes
save_model(model, model_save_path, classes)
print(f"Model saved to {model_save_path}")
