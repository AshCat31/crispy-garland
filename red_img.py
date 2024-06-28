import os

from PIL import Image


def process_image(input_image_path, output_image_path, red_threshold=235):
    """
    Process an image to threshold the red channel and save the processed image.

    Args:
    - input_image_path: Path to the input image.
    - output_image_path: Path to save the processed image.
    - red_threshold: Threshold value for the red channel (default: 235).
    """
    # Open the image
    img = Image.open(input_image_path)

    # Convert the image to RGB mode (if not already in RGB mode)
    img = img.convert('RGB')

    # Get the image dimensions
    width, height = img.size

    # Create a blank image for the output
    output_img = Image.new('RGB', (width, height))

    # Process each pixel
    for x in range(width):
        for y in range(height):
            # Get pixel RGB values
            r, g, b = img.getpixel((x, y))

            # Check if red channel value is above threshold
            if r > red_threshold:
                # Set pixel to white
                output_img.putpixel((x, y), (255, 255, 255))
            else:
                # Keep original pixel color
                output_img.putpixel((x, y), (r, g, b))

    # Save the modified image
    output_img.save(output_image_path)


def process_images_from_file(ids_file):
    """
    Process images based on IDs listed in a file.

    Args:
    - ids_file: Path to the file containing IDs (each ID on a new line).
    """
    with open(ids_file, 'r') as f:
        for line in f:
            # Clean the ID (remove any extra whitespace)
            id = line.strip()

            # Construct paths for input and output images
            input_image_path = f"/home/canyon/S3bucket/{id}/6_inch.png"
            output_image_path = f"/home/canyon/S3bucket/{id}/6_inch.png"
            red_output_image_path = f"/home/canyon/Test_Equipment/crispy-garland/red/{id}red_6_inch.png"

            # Save original as red_6_inch.png
            os.rename(input_image_path, red_output_image_path)

            # Process the image
            process_image(red_output_image_path, output_image_path)


# Example usage:
ids_file = 'QA_ids.txt'
process_images_from_file(ids_file)
