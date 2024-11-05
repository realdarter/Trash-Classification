from image_classifier import *

# Example usage to predict an image
image_path = r"C:\Users\minec\Downloads\test\metal.jpg"  # Replace with your image path
image_tensor = preprocess_image(image_path)

model, classes = load_model("checkpoint/save1")

top_k_predictions = predict(model, image_tensor, classes, k=3)
for class_name, prob in top_k_predictions:
    print(f"Class {class_name}: {prob:.2f}%")