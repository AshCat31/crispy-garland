"""This module is responsible for communication with s3 bucket"""

import io
import json

import cv2
import numpy as np

from s3_setup import S3Setup

IMAGE_SIZE = (240, 320)
CALIBRATION_POINTS_LENGTH = 9
JSON_DEVICE_TYPE = "device_type"
JSON_CAMERA_ID = "camera_id"
HYDRA_DEVICE_NAME = "hydra"
HUB_DEVICE_NAME = "hub"
HUB_MOSAIC_NAME = "mosaic"

s3s = S3Setup()
s3client, BUCKET_NAME = s3s()


def get_file_from_s3(filename: str):
    """Download file from s3 as bytes

    Args:
        filename (str): list of coordinates to be converted

    Returns:
        body of response in io.BytesIO format
    """
    raw_response = s3client.get_object(Bucket=BUCKET_NAME, Key=filename)
    raw_bytes = io.BytesIO(raw_response["Body"].read())
    raw_bytes.seek(0)
    return raw_bytes


def load_rgb_image_from_s3(filename: str):
    """Download rgb image from s3"""
    raw_image_bytes = get_file_from_s3(filename)
    raw_image_np_bytes = np.asarray(bytearray(raw_image_bytes.read()), dtype=np.uint8)
    rgb_image = cv2.imdecode(raw_image_np_bytes, cv2.IMREAD_COLOR)
    if np.max(rgb_image) <= 1:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    return rgb_image


def load_numpy_array_from_s3(filename: str):
    """Download numpy array from s3"""
    raw_numpy_bytes = get_file_from_s3(filename)
    numpy_array = np.load(raw_numpy_bytes)
    return numpy_array


def load_thermal_image_from_s3(device_id: str):
    """Download thermal image from s3"""
    thermal_image = load_numpy_array_from_s3(f"{device_id}/6_inch.npy")
    thermal_image = np.transpose(thermal_image)
    thermal_image_scaled = cv2.resize(thermal_image, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    thermal_image_scaled = np.array(
        (thermal_image_scaled - np.min(thermal_image_scaled))
        / (np.max(thermal_image_scaled) - np.min(thermal_image_scaled))
        * 255
    ).astype(np.uint8)
    return thermal_image_scaled


def load_json_from_s3(filename):
    """Load json file from s3"""
    raw_json_bytes = get_file_from_s3(filename)
    return json.load(raw_json_bytes)


def write_json_to_s3(filename, json_data):
    """Load json file from s3"""
    buffer_array = io.BytesIO()
    buffer_array.write(json.dumps(json_data).encode())
    buffer_array.seek(0)
    write_file_to_s3(filename, buffer_array)


def get_device_type_and_idx(device_id: str):
    """Download device information from s3"""
    json_data = load_json_from_s3(f"{device_id}/data.json")
    if JSON_DEVICE_TYPE in json_data:
        device_type = json_data[JSON_DEVICE_TYPE]
        if device_type == HUB_DEVICE_NAME:
            device_type = HUB_MOSAIC_NAME
    else:
        device_type = "error"

    if device_type == HYDRA_DEVICE_NAME or JSON_CAMERA_ID not in json_data:
        device_idx = device_id
    else:
        device_idx = json_data[JSON_CAMERA_ID]

    return device_type, device_idx


def update_data_json_on_s3(device_id, data_array):
    """Write new entries or update existing in s3 data.json"""
    filename = f"{device_id}/data.json"
    json_data = load_json_from_s3(filename)
    for key, value in data_array:
        json_data[key] = value
    write_json_to_s3(filename, json_data)


def coords_to_array(coordinates: list[(int, int)]):
    """Convert normal list to numpy array"""
    numpy_arr = np.zeros((len(coordinates), 2))
    for i, (x, y) in enumerate(coordinates):
        numpy_arr[i][0] = int(x)
        numpy_arr[i][1] = int(y)
    return numpy_arr


def write_file_to_s3(filename: str, file: io.BytesIO):
    """Write file from s3 as bytes

    Args:
        filename (str): list of coordinates to be converted
        file (io.BytesIO): file in binary format
    """
    s3client.upload_fileobj(Fileobj=file, Bucket=BUCKET_NAME, Key=filename)


def write_numpy_to_s3(filename, array):
    """Write numpy array to s3 as .npy file"""
    buffer_array = io.BytesIO()
    np.save(buffer_array, array)
    buffer_array.seek(0)
    write_file_to_s3(filename, buffer_array)


def write_image_to_s3(filename, image):
    """Write image to s3 as .png file"""
    is_success, buffer = cv2.imencode(".png", image)
    if is_success:
        write_file_to_s3(filename, io.BytesIO(buffer))
