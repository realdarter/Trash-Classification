import tkinter as tk
from tkinter import filedialog
from image_classifier import *


def choose_image():
    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*"))
    )
    return img_path

model_dir = 'saved_models/save1'

args = create_args(
        num_epochs=2, 
        batch_size=32, 
        learning_rate=1e-5, 
        save_every=500, 
        max_length=256, 
        temperature=1.0, 
        top_k=1, #adjust for the top classified types
        top_p=0.9, 
        repetition_penalty=1.0
        )

model, epoch, classes = load_model(model_dir)
while True:
    #test_image_path = r'C:\Users\minec\Downloads\test\metal.jpg'
    test_image_path = choose_image()
    top_k_labels, top_k_probs = predict_images(img_path=test_image_path, model=model, classes=classes, args=args)
    print(top_k_labels, top_k_probs)
