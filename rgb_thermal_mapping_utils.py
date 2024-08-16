"""This module is responsible for functions that are used by rgb-thermal mapping functions"""

import cv2
import numpy as np

IMAGE_SIZE = (240, 320)
RECTANGLE_START = (100, 100)
RECTANGLE_END = (340, 420)
RECTANGLE_COLOR = (0, 0, 255)
RECTANGLE_THICKNESS = 3


def generate_parallax_image(parallax_mask):
    """Generate image representing parallax mask and rgb camera fov"""
    mask_img = parallax_mask.astype(np.uint8) * 255
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(mask_img, RECTANGLE_START, RECTANGLE_END, RECTANGLE_COLOR, RECTANGLE_THICKNESS)
    mask_img = cv2.resize(mask_img, IMAGE_SIZE)
    return mask_img


def generate_debug_image(thermal_image, thermal_coordinates, rgb_image, rgb_coordinates, parallax_mask):
    """Generate debug image that shows result of calibration"""
    debug_thermal_image = thermal_image.copy()
    debug_rgb_image = rgb_image.copy()

    if debug_thermal_image.ndim != 3:
        debug_thermal_image = cv2.cvtColor(debug_thermal_image, cv2.COLOR_GRAY2RGB)

    for x, y in thermal_coordinates:
        cv2.circle(debug_thermal_image, (int(x), int(y)), 0, (0, 0, 255), 10)

    for x, y in rgb_coordinates:
        cv2.circle(debug_rgb_image, (int(x), int(y)), 0, (0, 0, 255), 10)

    debug_image = np.concatenate((debug_thermal_image, debug_rgb_image), axis=1)
    debug_image = np.concatenate((debug_image, generate_parallax_image(parallax_mask)), axis=1)
    return debug_image
