import cv2
import numpy as np

import os
import json
import math

CALIBRATION_DATA_PATH = "/home/jacek/delta-thermal/calibration_data"


def get_mask_filename(base_path, device_id):
    """Return filename for mask file based on device_id"""
    unit_type = "hydra_" if device_id[0] == 'E' else "mosaic_"

    if unit_type == "hydra_":
        img_idx = device_id
    else:
        # find camera ID
        try:
            with open(os.path.join(base_path, 'data.json')) as f:
                data = json.load(f)
                img_idx = data['camera_id']
        except FileNotFoundError:
            return ""

    return f"{base_path}/calculated_transformations/{img_idx}/mapped_mask_matrix_{unit_type}{img_idx}.npy"


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


def validate_mask(mask):
    mask_8u = np.uint8(mask)

    contours, _ = cv2.findContours(
        mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 1:
        cnt = contours[0]
        perimeter = cv2.arcLength(cnt,True)
        hull = cv2.convexHull(cnt)
        hull_perimeter = cv2.arcLength(hull,True)
        perimeter_difference = perimeter - hull_perimeter
        return perimeter_difference < 500

    return False


if __name__ == "__main__":
    devices_list = try_to_read_list_from_file(
        "mask_validator/calibrated_devices.txt")
    total = len(devices_list)
    incorrect = 0
    counter = 0
    for device_id in devices_list:
        mask_path = get_mask_filename(
            f"{CALIBRATION_DATA_PATH}/{device_id}", device_id)
        if mask_path != "":
            try:
                mask = np.load(mask_path)
                if validate_mask(mask) is False:
                    print(device_id)
                    incorrect += 1
            except FileNotFoundError:
                pass

    print(f"Total: {total}, correct: {total-incorrect}, incorrect: {incorrect}")
