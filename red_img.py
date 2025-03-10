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
    img = Image.open(input_image_path)
    img = img.convert("RGB")
    width, height = img.size

    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x, y))
            if r > red_threshold:
                img.putpixel((x, y), (255, 255, 255))

    img.save(output_image_path)


def process_images_from_file(ids_file):
    """
    Process images based on IDs listed in a file.

    Args:
    - ids_file: Path to the file containing IDs (each ID on a new line).
    """
    with open(ids_file, "r") as f:
        for line in f:
            id = line.split()[0]

            input_image_path = f"/home/canyon/S3bucket/{id}/6_inch.png"
            output_image_path = f"/home/canyon/S3bucket/{id}/6_inch.png"
            red_output_image_path = f"/home/canyon/Test_Equipment/crispy-garland/red/{id}red_6_inch.png"

            # Save original as red_6_inch.png
            os.rename(input_image_path, red_output_image_path)
            process_image(red_output_image_path, output_image_path)  # must pass red path bcuz input was moved


ids_file = "QA_ids.txt"
process_images_from_file(ids_file)
