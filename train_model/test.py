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
# basically using the while loop. May not work if image path is different from the image itself.
def predictImage(image):
    return predict_images(image, model=model, classes=classes, args=args)
model, epoch, classes = load_model(model_dir)
