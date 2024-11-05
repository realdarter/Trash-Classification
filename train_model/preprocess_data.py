from PIL import Image
import os

def resize_or_crop_image(input_path, target_size=(512, 384)):
    """
    Resizes or crops an image to the target size while maintaining aspect ratio.
    input_path (str): Path to the input image file.
    target_size (tuple): Desired dimensions (width, height) for the output image.
    Returns: Image: Resized or cropped PIL Image.
    """
    try:
        with Image.open(input_path) as img:
            img.thumbnail((target_size[0], target_size[1]), Image.ANTIALIAS)

            left = (img.width - target_size[0]) / 2
            top = (img.height - target_size[1]) / 2
            right = (img.width + target_size[0]) / 2
            bottom = (img.height + target_size[1]) / 2

            img = img.crop((left, top, right, bottom)) if (img.width > target_size[0] or img.height > target_size[1]) else img

            return img
    except Exception as e:
        print(f"Error resizing/cropping {input_path}: {e}")
        return None

def convert_to_jpg(input_path, output_path=None, quality=85):
    """
    Converts an image to JPEG format.
    input_path (str): Path to the input image file.
    output_path (str): Path to save the JPEG file. If None, saves as the same name with .jpg extension.
    quality (int): Quality of the saved JPEG image (1 to 100). Defaults to 85 for good balance.
    Returns: str: Path to the converted JPEG file.
    """
    try:
        # Resize or crop the image first
        img = resize_or_crop_image(input_path)
        if img is None:
            return None  # Return if resizing/cropping failed

        # Convert image to RGB (JPEG does not support transparency)
        img = img.convert("RGB")

        if output_path is None:
            base = os.path.splitext(input_path)[0]
            output_path = f"{base}.jpg"

        img.save(output_path, "JPEG", quality=quality)
        print(f"Image saved as {output_path}")

        return output_path
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return None