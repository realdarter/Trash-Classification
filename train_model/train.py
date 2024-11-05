from image_classifier import *  # Ensure this imports the necessary functions and classes

data_dir = 'data/garbage_classification'
model_save_path = 'checkpoint/save1'
batch_size = 100
# Load data
train_loader, classes = load_data(data_dir, batch_size)

# Create arguments for training
args = create_args(num_epochs=16, 
                   batch_size=batch_size,
                   )

model = SimpleCNN(num_classes=len(classes)).to(device)

train_model(model_save_path, data_dir, train_loader, args)

save_model(model, model_save_path, classes)
print(f"Model saved to {model_save_path}")

test_model(model_save_path, data_dir, args)