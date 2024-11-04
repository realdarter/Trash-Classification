from PIL import Image
import os

def convert_to_jpg(input_path, output_path=None, quality=85):
    """
    Converts an image to JPEG format.
    input_path (str): Path to the input image file.
    output_path (str): Path to save the JPEG file. If None, saves as the same name with .jpg extension.
    quality (int): Quality of the saved JPEG image (1 to 100). Defaults to 85 for good balance.
    Returns: str: Path to the converted JPEG file.
    """
    try:
        # Open the image file
        with Image.open(input_path) as img:
            # Convert image to RGB (JPEG does not support transparency)
            img = img.convert("RGB")
            
            # Define the output path if not specified
            if output_path is None:
                base = os.path.splitext(input_path)[0]
                output_path = f"{base}.jpg"
                
            # Save the image as JPEG
            img.save(output_path, "JPEG", quality=quality)
            print(f"Image saved as {output_path}")
            
            return output_path
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return None