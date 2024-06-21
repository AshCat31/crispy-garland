"""This module is responsible for user-interactive calibration"""

import os
import logging

import numpy as np
import cv2
from matplotlib import pyplot as plt

import automatic_calibration

from paralax_calibrator import ParalaxCalibrator

from automatic_point_detection import calibration_utils

import s3_utils

from rgb_thermal_mapping_utils import generate_debug_image

PADDED_IMAGE_SIZE = (520, 440)
SCALED_IMAGE_SIZE = (240, 320)
VALIDATION_TEXT = "y - accept | n - reject | q - quit_accept | c - quit_reject | r - repeat"


def manual_sample_coordinate_of_corners(img, num_corners=9, padding=0):
    """ 
    The function plots the image and enable sampling (using mouse clicks) 9 (x,y) values
    The sampling should be preformed in the following order:
    0     1     2

    3     4     5

    6     7     8
    The function returns numpy array of size (9,2) for 9 samples of x,y values
    If the sampling process failed, the function returns None 
    """
    print("sample ", num_corners, "in image using ginput")

    if padding > 0:
        if np.ndim(img) == 2:
            img_height, img_width = np.shape(img)
            padded_img = np.zeros(
                (img_height+2*padding, img_width + 2*padding)).astype(np.uint8)
            padded_img[padding:padding+img_height,
                       padding:padding+img_width] = img[:, :]
        elif np.ndim(img) == 3:
            img_height, img_width, n_cnl = np.shape(img)
            padded_img = np.zeros(
                (img_height + 2 * padding, img_width + 2 * padding, n_cnl)).astype(np.uint8)
            padded_img[padding:padding + img_height,
                       padding:padding + img_width, :] = img[:, :, :]
        else:
            logging.warning("Can't pad image, ndim not standard")
            return None
        img = padded_img

    plt.imshow(img, cmap='gray')
    samples_coord = plt.ginput(num_corners, timeout=90)
    plt.show()

    if len(samples_coord) == num_corners:
        # Convert to np array
        coord_arr = np.zeros((num_corners, 2))
        for i in range(num_corners):
            coord_arr[i][0] = samples_coord[i][0] - padding
            coord_arr[i][1] = samples_coord[i][1] - padding

        return coord_arr

    logging.warning("Failed to sample "+str(num_corners)+" coordinates")
    return None


def save_sampled_coordinates(coord_arr, file_name, use_s3):
    """Saving the sampled coordinates"""
    if use_s3:
        s3_utils.write_numpy_to_s3(file_name, coord_arr)
    else:
        np.save(file_name, coord_arr)


def load_sampled_coordinates(file_name, use_s3):
    """Loading the sampled coordinates"""
    if use_s3:
        coord_arr = s3_utils.load_numpy_array_from_s3(file_name)
    else: 
        coord_arr = np.load(file_name)
    return coord_arr


def save_rgb_thermal_mapping_to_fs(output_folder, img_idx, unit_type, coordinate_map, mask_martix, sensitivity_matrix, debug_image):
    """Save rgb thermal mapping on disc"""
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(output_folder+'/calculated_transformations/'):
        os.mkdir(output_folder+'/calculated_transformations/')
    if not os.path.exists(output_folder+'/calculated_transformations/'+img_idx):
        os.mkdir(output_folder+'/calculated_transformations/'+img_idx)
    np.save(output_folder+'/calculated_transformations/'+img_idx +
            '/mapped_coordinates_matrix_'+unit_type+img_idx+'.npy', coordinate_map)
    np.save(output_folder+'/calculated_transformations/'+img_idx +
            '/mapped_mask_matrix_'+unit_type+img_idx+'.npy', mask_martix)
    np.save(output_folder+'/calculated_transformations/'+img_idx +
            '/sensitivity_correction_matrix_'+unit_type+img_idx+'.npy', sensitivity_matrix)
    cv2.imwrite(output_folder + "/debug_image.png", debug_image)
    

def save_rgb_thermal_mapping_to_s3(device_id, img_idx, unit_type, coordinate_map, mask_martix, sensitivity_matrix, debug_image):
    """Save rgb thermal mapping on s3"""
    directory = f"{device_id}/calculated_transformations/{img_idx}"
    s3_utils.write_numpy_to_s3(
        f"{directory}/mapped_coordinates_matrix_{unit_type}_{img_idx}.npy", coordinate_map)
    s3_utils.write_numpy_to_s3(
        f"{directory}/mapped_mask_matrix_{unit_type}_{img_idx}.npy", mask_martix)
    s3_utils.write_numpy_to_s3(
        f"{directory}/sensitivity_correction_matrix_{unit_type}_{img_idx}.npy", sensitivity_matrix)
    s3_utils.write_image_to_s3(f"{device_id}/debug_image.png", debug_image)


def sample_coordinates(calibration_image, coordinates_file_path, is_image_thermal, use_s3, is_hydra):
    """Samle coordinates of calibration points"""
    if is_image_thermal:
        coordinates = automatic_calibration.detect_calibration_points_thermal_image(
            calibration_image, is_hydra)
    else:
        coordinates = automatic_calibration.detect_calibration_points_rgb_image(
            calibration_image)

    if coordinates is None:
        # Manually get the 9 (x,y) coordinates of the 9 elements and save them
        coordinates = manual_sample_coordinate_of_corners(
            calibration_image)  # thermal element coordinates

    if coordinates.all():
        save_sampled_coordinates(coordinates, coordinates_file_path, use_s3)
        return coordinates
    return None


def manual_validation(img_idx, debug_image):
    """Let the user see images and decide if they are correct or not"""
    window_name = f"{img_idx} {VALIDATION_TEXT}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 640)
    cv2.imshow(window_name, debug_image)
    key = cv2.waitKey(20000)
    cv2.destroyAllWindows()

    logging.debug(f"Pressed key = {key}")

    return key


def calculate_rgb_thermal_mapping(img_idx, rgb_img_path, thermal_img_path, rgb_coordinates_file_path, thermal_coordinates_file_path, thermal_flip_lr=False, thermal_flip_ud=False, ginput_padding=0, output_folder=None, rotate_rgb=False, unit_type='', use_s3=False, device_id = ""):
    """
    The function generates new transformation from thermal image to RGB image
    using a 9-elements calibaration board


    :param img_idx: Index of the (RGB,thermal) images
    :param rgb_img_path: Path to the rgb image
    :param thermal_img_path: Path to the thermal numpy array (.npy). Can be 2D (single frame) or 3D (named 'imglist',
           a sequence of thermal frames)
    :param thermal_flip_lr: If need to flip the termal image left to right
    :param thermal_flip_ud: If need to flip the termal image upside down
    :return: The function saved the transformation matrices (2 matrices - transformation and mask) as .npy files
    """

    if use_s3:
        thermal_img = s3_utils.load_thermal_image_from_s3(device_id)
        rgb_img = s3_utils.load_rgb_image_from_s3(f'{device_id}/6_inch.png')
        rgb_coordinates_file_path = f"{device_id}/rgb_{img_idx}_9element_coord.npy"
        thermal_coordinates_file_path = f"{device_id}/trml_{img_idx}_9element_coord.npy"
    else:
        thermal_img = calibration_utils.load_thermal_image(
            thermal_img_path, thermal_flip_ud, thermal_flip_lr)
        rgb_img = calibration_utils.load_rgb_image(rgb_img_path, rotate_rgb)

    # Select if you want to sample new coordinates on the RGB and thermal image,
    # or you want to use the saved values (is exist)
    sample_new_coordinates_flag_rgb = True
    sample_new_coordinates_flag_thermal = True

    is_hydra = img_idx == device_id

    while True:
        logging.debug("Calculating new transformation")

        if sample_new_coordinates_flag_rgb:
            logging.debug("Sampling new RGB coordinates")
            rgb_elements_coordinates = sample_coordinates(
                rgb_img, rgb_coordinates_file_path, False, use_s3, is_hydra)
        else:
            # Load saved values
            logging.debug("Using exist RGB coordinates")
            rgb_elements_coordinates = load_sampled_coordinates(
                rgb_coordinates_file_path, use_s3)

        if sample_new_coordinates_flag_thermal:
            logging.debug("Sampling new thermal coordinates")
            thermal_elements_coordinates = sample_coordinates(
                thermal_img, thermal_coordinates_file_path, True, use_s3, is_hydra)
        else:
            # Load saved values
            logging.debug("Using exist thermal coordinates")
            thermal_elements_coordinates = load_sampled_coordinates(
                thermal_coordinates_file_path, use_s3)

        logging.debug("Statring to calculate transformation matrices")

        paralax_calibrator = ParalaxCalibrator()
        mask_martix, coordinate_map, sensitivity_matrix = paralax_calibrator(
            thermal_img,
            rgb_img,
            thermal_elements_coordinates,
            rgb_elements_coordinates
        )
        logging.debug(mask_martix.shape)
        logging.debug(coordinate_map.shape)
        logging.debug(sensitivity_matrix.shape)

        debug_image = generate_debug_image(
            thermal_img, thermal_elements_coordinates, rgb_img, rgb_elements_coordinates, mask_martix)

        key = manual_validation(img_idx, debug_image)

        if key == ord('y') or key == ord('q'):
            if use_s3:
                save_rgb_thermal_mapping_to_s3(
                    device_id, img_idx, unit_type, coordinate_map, mask_martix, sensitivity_matrix, debug_image)
            else:
                save_rgb_thermal_mapping_to_fs(
                    output_folder, img_idx, unit_type, coordinate_map, mask_martix, sensitivity_matrix, debug_image)
            return True, key == ord('q')  # This device is correct, quit if 'q'

        if key == ord('n'):
            return False, False  # This device is corrupted, do not quit

        if key == ord('c'):
            return False, True  # This device is corrupted, quit

        logging.debug("Calibrating image again")


if __name__ == "__main__":
    calculate_rgb_thermal_mapping("E66138528342A835", "/home/jacek/delta-thermal/calibration_data/E66138528342A835/6_inch.png", "/home/jacek/delta-thermal/calibration_data/E66138528342A835/6_inch.npy",
                                  "/home/jacek/delta-thermal/calibration_data/E66138528342A835/rgb_E66138528342A835_9element_coord.npy",
                                  "/home/jacek/delta-thermal/calibration_data/E66138528342A835/trml_E66138528342A835_9element_coord.npy",
                                  False, False, output_folder="/home/jacek/delta-thermal/calibration_data/E66138528342A835", rotate_rgb=False, unit_type="hydra_", use_s3=False, device_id="E66138528342A835")