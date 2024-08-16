"""A set of utils for automatic calibration development"""

import os
import json
import logging

import cv2
import numpy as np

CALIBRATION_DATA_PATH = "/home/jacek/delta-thermal/calibration_data/"


def load_thermal_image(path: str, thermal_flip_ud: bool = False, thermal_flip_lr: bool = False):
    """Loads thermal image the same way delta-thermal-rgb-mapping does.

    Args:
        path (str): path thermal calibration image
        thermal_flip_ud (bool): should be calibration image flipped along axis 0
        thermal_flip_lr (bool): should be calibration image flipped along axis 1

    Returns:
        trml_matrix_scaled - thermal image that is already transposed and scaled
    """
    trml_arr = np.load(path)
    if trml_arr.ndim == 3:
        trml_arr = np.mean(trml_arr, axis=0)

    if thermal_flip_ud:
        trml_arr = np.flipud(trml_arr)
    if thermal_flip_lr:
        trml_arr = np.fliplr(trml_arr)

    trml_arr = np.transpose(trml_arr)
    # Scale the thermal image by a factor or 10 (from 32x24 to 320x240)
    # for better smoothness of the heatmap
    dest_size = (240, 320)
    trml_matrix_scaled = cv2.resize(trml_arr, dest_size, interpolation=cv2.INTER_CUBIC)
    trml_matrix_scaled = np.array(
        (trml_matrix_scaled - np.min(trml_matrix_scaled)) / (np.max(trml_matrix_scaled) - np.min(trml_matrix_scaled)) * 255
    ).astype(np.uint8)

    return trml_matrix_scaled


def load_rgb_image(path: str, rotate_rgb: bool = False):
    """Loads rbg image the same way delta-thermal-rgb-mapping does.

    Args:
        path (str): path rgb calibration image
        rotate_rgb (bool): should rgb image be rotated by 90 degrees

    Returns:
        trml_matrix_scaled - thermal image that is already transposed and scaled
    """
    rgb_img = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    if rotate_rgb:
        rgb_img = np.rot90(rgb_img)

    # If image loaded in [0,1] values, convert it to [0,255] uint8 format
    if np.max(rgb_img) <= 1:
        rgb_img = (rgb_img * 255).astype(np.uint8)

    return rgb_img


def try_to_read_list_from_file(filename):
    """Tries to load data from file. If no file, returns empty array.

    Args:
        filename (str): path to filename

    Returns:
        data - array of stripped lines from file
    """
    data = []
    try:
        with open(filename) as list_from_file:
            data = [line.strip() for line in list_from_file]
    except FileNotFoundError:
        pass
    return data


def load_coordinates(path):
    """Load correct coordinates from path, scale them if they are unscaled"""
    calibration_points = np.load(path)
    if calibration_points.max() <= 32:
        calibration_points *= 10

    calibration_points = np.array(calibration_points).astype(np.uint16)

    return calibration_points


def get_coordinates_filename(base_path, device_id):
    """Return filename for coordinates file based on device_id"""
    unit_type = "hydra_" if device_id[0] == "E" else "mosaic_"

    if unit_type == "hydra_":
        img_idx = device_id
    else:
        # find camera ID
        try:
            with open(os.path.join(base_path, "data.json")) as f:
                data = json.load(f)
                img_idx = data["camera_id"]
            if img_idx is None:
                logging.warning(f"no camera id found for {device_id}")
        except FileNotFoundError:
            logging.error(f"no 'data.json' file for device {device_id}")
            return ""

    return f"trml_{img_idx}_9element_coord.npy"


def get_data_for_device(device_id):
    """Load thermal image and correct calibration points for selected device"""
    base_path = os.path.join(CALIBRATION_DATA_PATH, device_id)
    trml_img_path = os.path.join(base_path, "6_inch.npy")

    coordinates_filename = get_coordinates_filename(base_path, device_id)
    if len(coordinates_filename) == 0:  # No filename, error
        return False, None, None
    trml_coordinates_file_path = os.path.join(base_path, coordinates_filename)

    try:
        calibration_points = load_coordinates(trml_coordinates_file_path)
        thermal_image = load_thermal_image(trml_img_path)
    except FileNotFoundError:
        return False, None, None

    return True, calibration_points, thermal_image


def get_device_calibration_images(device_id):
    """Load thermal image and rgb image for selected device"""
    base_path = os.path.join(CALIBRATION_DATA_PATH, device_id)
    thermal_img_path = os.path.join(base_path, "6_inch.npy")
    rgb_img_path = os.path.join(base_path, "6_inch.png")

    try:
        thermal_image = load_thermal_image(thermal_img_path)
        rgb_image = load_rgb_image(rgb_img_path)
    except FileNotFoundError:
        return False, None, None

    return True, thermal_image, rgb_image
