from image_classifier import *

# Example usage to predict an image
image_path = r"C:\Users\minec\Downloads\test\metal.jpg"  # Replace with your image path


test_image_path = r'C:\Users\minec\Downloads\test\metal.jpg'  # Replace with your image path
    predict_top_k(test_image_path, model, transformations, args, device=device)