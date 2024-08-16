"""Print images of automatic thermal calibration into specified directories"""

import os
import cv2
import numpy as np

import auto_point_detection

import calibration_utils


CALIBRATION_DATA_PATH = "/home/jacek/delta-thermal/calibration_data/"
THERMAL_TESTS_DIRECTORY = "thermal_tests_dataset"
OUTPUT_DIR = "generated_images"


def print_images(devices: list[str], dir_name: str):
    """Print test images into selected directory

    Args:
        image_list (list[str]): list of images to be printed
        dir_name (str): directory for test images
    """
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    for device_id in devices:
        success, correct_points, thermal_image = calibration_utils.get_data_for_device(device_id)
        if not success:
            continue

        _, thermal_rgb, thresh_rgb = auto_point_detection.find_calibration_points_on_heatmap(thermal_image)

        thermal_rgb = np.array((thermal_rgb - np.min(thermal_rgb)) / (np.max(thermal_rgb) - np.min(thermal_rgb)) * 255).astype(
            np.uint8
        )

        base_path = os.path.join(CALIBRATION_DATA_PATH, device_id)
        rgb_img_path = os.path.join(base_path, "6_inch.png")
        rgb_image = cv2.imread(rgb_img_path)

        for x, y in correct_points:
            cv2.circle(thermal_rgb, (x, y), 0, (0, 255, 0), 10)

        result = np.concatenate((rgb_image, thermal_rgb), axis=1)
        result = np.concatenate((result, thresh_rgb), axis=1)

        cv2.imwrite(f"{dir_name}/{device_id}.png", result)


def thermal_print_images():
    """Print correct and corrupted images in selective directories"""
    test_data_dir = os.path.join(os.getcwd(), THERMAL_TESTS_DIRECTORY)
    output_dir = os.path.join(os.getcwd(), OUTPUT_DIR)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    filenames = ["correct.txt", "errors.txt", "failures.txt"]

    for filename in filenames:
        file_path = os.path.join(test_data_dir, filename)
        devices = calibration_utils.try_to_read_list_from_file(file_path)

        output_file_dir = os.path.join(output_dir, filename.split(".", maxsplit=1)[0])
        print_images(devices, output_file_dir)


if __name__ == "__main__":
    thermal_print_images()
