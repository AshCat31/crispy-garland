import sys
from PIL import Image

def print_image_dimensions(image_path):
    try:
        # Open an image file
        with Image.open(image_path) as img:
            # Get image dimensions
            width, height = img.size
            print(f"Image: {image_path}")
            print(f"Dimensions: {width}x{height}")
    
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: drag and drop one or more image files onto this script.")
    else:
        for file_path in sys.argv[1:]:
            print_image_dimensions(file_path)
