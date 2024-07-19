"""This module is responsible for automatic thermal mapping"""
import logging
import os

import cv2

from paralax_calibrator import ParalaxCalibrator

from automatic_point_detection import auto_point_detection
from rgb_thermal_mapping_utils import generate_debug_image

from s3_utils import *


class Mapper:
    def __init__(self, s3client=None, bucket_name=None):
        self.IMAGE_SIZE = (240, 320)
        self.CALIBRATION_POINTS_LENGTH = 9
        self.JSON_DEVICE_TYPE = 'device_type'
        self.JSON_CAMERA_ID = 'camera_id'
        self.HYDRA_DEVICE_NAME = "hydra"
        self.HUB_DEVICE_NAME = "hub"
        self.HUB_MOSAIC_NAME = "mosaic"
        self.MAX_PERIMETER_DIFFERENCE = 500
        self.errors = []

        # Setup boto3
        cred = boto3.Session().get_credentials()
        ACCESS_KEY = cred.access_key
        SECRET_KEY = cred.secret_key
        SESSION_TOKEN = cred.token

        if s3client is None:
            self.s3client = boto3.client('s3',
                                aws_access_key_id=ACCESS_KEY,
                                aws_secret_access_key=SECRET_KEY,
                                aws_session_token=SESSION_TOKEN,
                                )
            self.BUCKET_NAME = 'kcam-calibration-data'
        else:
            self.s3client = s3client
            self.BUCKET_NAME = bucket_name


    def see_image(self, image: cv2.Mat, device_id: str):
        """Debug function to display image"""
        window_name = f"{device_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 640)
        cv2.moveWindow(window_name, 0, 0)
        cv2.imshow(window_name, image)
        # cv2.waitKey(20000)
        # cv2.destroyAllWindows()
        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
            if cv2.waitKey(100) > 0:
                break


    def validate_calibration_points(self, calibration_points: str):
        """Check if number of calibration points is correct"""
        return len(calibration_points) == self.CALIBRATION_POINTS_LENGTH


    def validate_mask_matrix(self, mask_matrix):
        """Check if mask matrix is correct by checking if there is only one shape"""
        mask_8u = np.uint8(mask_matrix)

        contours, _ = cv2.findContours(
            mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 1:
            cnt = contours[0]
            perimeter = cv2.arcLength(cnt, True)
            hull = cv2.convexHull(cnt)
            hull_perimeter = cv2.arcLength(hull, True)
            perimeter_difference = perimeter - hull_perimeter
            return perimeter_difference < self.MAX_PERIMETER_DIFFERENCE

        return False


    def convert_coordinates_to_numpy_array(self, coordinates: list[(int, int)]):
        """Convert normal list to numpy array"""
        numpy_arr = np.zeros((len(coordinates), 2))
        for i, (x, y) in enumerate(coordinates):
            numpy_arr[i][0] = int(x)
            numpy_arr[i][1] = int(y)
        return numpy_arr

    def brighten_corner(self, image_array):
        height, width = image_array.shape[:2]

        # Calculate boundaries for each corner
        start_row1 = 0
        end_row1 = height // 3
        start_col1 = 0
        end_col1 = width // 3

        start_row2 = 0
        end_row2 = height // 3
        start_col2 = 2 * width // 3
        end_col2 = width

        start_row3 = 2 * height // 3
        end_row3 = height
        start_col3 = 0
        end_col3 = width // 3

        start_row4 = 4 * height // 5  # GO BACK TO QUARTER
        end_row4 = height
        start_col4 = 4 * width // 5
        end_col4 = width

        # Extract corners
        corner1 = image_array[start_row1:end_row1, start_col1:end_col1]
        corner2 = image_array[start_row2:end_row2, start_col2:end_col2]
        corner3 = image_array[start_row3:end_row3, start_col3:end_col3]
        corner4 = image_array[start_row4:end_row4, start_col4:end_col4]

        # Find minimum brightness values in each corner
        min_brightness1 = np.min(corner1)
        min_brightness2 = np.min(corner2)
        min_brightness3 = np.min(corner3)
        min_brightness4 = np.min(corner4)
        max_brightness4 = np.max(corner4)
        normalized_corner4 = (corner4 - min_brightness4) / (max_brightness4 - min_brightness4)
        contrast_factor = 1.5  # Adjust as needed
        # Calculate brightness adjustments based on differences from min_brightness
        brightness_adjustment1 = 1 + 0.06 *0.06* (corner1 - min_brightness1)
        brightness_adjustment2 = 1 + 0.06 *0.06* (corner2 - min_brightness2)
        brightness_adjustment3 = 1 + 0.06 *0.06* (corner3 - min_brightness3)
        brightness_adjustment4 = 1 + 0.08 *0.07* (corner4 - min_brightness4)

        adjusted_corner4 = (normalized_corner4 - 0.5) * contrast_factor + 0.5
        adjusted_corner4 = adjusted_corner4 * (max_brightness4 - min_brightness4) + min_brightness4

        # Ensure values are within [0, 255] range
        brightened_corner1 = np.clip(corner1 * brightness_adjustment1, 0, 255).astype(np.uint8)
        brightened_corner2 = np.clip(corner2 * brightness_adjustment2, 0, 255).astype(np.uint8)
        brightened_corner3 = np.clip(corner3 * brightness_adjustment3, 0, 255).astype(np.uint8)
        # brightened_corner4 = np.clip(corner4 * brightness_adjustment4, 0, 255).astype(np.uint8)
        brightened_corner4 = np.clip(adjusted_corner4, 0, 255).astype(np.uint8)  # this method, only fails 3 of br, previously 20+

        brightened_image = np.copy(image_array)
        # brightened_image[start_row1:end_row1, start_col1:end_col1] = brightened_corner1  # tl
        # brightened_image[start_row2:end_row2, start_col2:end_col2] = brightened_corner2  # tr?
        # brightened_image[start_row3:end_row3, start_col3:end_col3] = brightened_corner3  # bl?
        brightened_image[start_row4:end_row4, start_col4:end_col4] = brightened_corner4  # br

###### 23/41 with (1/41 too bright), 11/41 w/o
        return brightened_image
    
    def contrast_by_sections(self, image_array, num_sections=3):
        height, width = image_array.shape[:2]
        section_height = height // num_sections
        section_width = width // num_sections
        brightened_image = np.copy(image_array)
        for i in range(num_sections):
            for j in range(num_sections):
                if (i,j) == (1,1):
                    # continue
                    pass
                contrast_factor = 1.3  # Adjust as needed
                start_row = i * section_height
                end_row = start_row + section_height
                start_col = j * section_width
                end_col = start_col + section_width
                section = image_array[start_row:end_row, start_col:end_col]

                min_brightness = np.min(section)
                max_brightness = np.max(section)

                normalized_section = (section - min_brightness) / (max_brightness - min_brightness)

                # Adjust contrast in the section using a contrast factor
                adjusted_section = (normalized_section - 0.5) * contrast_factor + 0.5

                # Denormalize the section back to the original brightness range
                adjusted_section = adjusted_section * (max_brightness - min_brightness) + min_brightness

                # Clip values to ensure they are within [0, 255]
                brightened_section = np.clip(adjusted_section, 0, 255).astype(np.uint8)

                # Replace the section in the brightened image
                brightened_image[start_row:end_row, start_col:end_col] = brightened_section

        return brightened_image

    def brighten_from_center(self, image):
        center_x = image.shape[0] // 2
        center_y = image.shape[1] // 2
        x_coords, y_coords = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        normalized_distances = distances / np.max(distances)
        brighter_image = image + (50*(normalized_distances.transpose())).astype(np.uint8)
        brighter_image = np.clip(brighter_image, 0, 255)
        return brighter_image
    
    def darken_in_center(self, image):
        center_x = image.shape[0] // 2
        center_y = image.shape[1] // 2
        x_coords, y_coords = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        normalized_distances = distances / np.max(distances)
        normalized_image = image / 255.0
        brightness_factor = 0.5 + 1.7 * normalized_distances  # Adjust the weights as desired
        adjusted_image = normalized_image * brightness_factor.transpose()
        adjusted_image = adjusted_image * 255.0
        adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
        return adjusted_image

    def contrast(self, image):
        normalized_image = image / 255.0
        contrast_factor = 1.2  # adjust as needed
        adjusted_image = (normalized_image - 0.5) * contrast_factor + 0.5
        adjusted_image = adjusted_image * 255.0
        adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
        return adjusted_image

    def do_automatic_rgb_calibration_mapping(self, device_id, debug_mode=True, overwrite=True):
        """Do automatic calibration mapping and send results to s3"""
        basePath = os.path.join("/home/canyon/S3bucket/", device_id)
        folder_path = basePath
        rgb_coordinates_file_path = folder_path + '/rgb_' + device_id + '_9element_coord.npy'
        trml_coordinates_file_path = folder_path + '/trml_' + device_id + '_9element_coord.npy'
        device_type, device_idx = get_device_type_and_idx(device_id)
        mask_exists = os.path.isfile(f"{device_id}/mapped_mask_matrix_{device_type}_{device_id}.npy")
        if mask_exists and not overwrite:
            print("exists")
            return "already exists"
        calibration_success = False
        thermal_image = self.brighten_corner(load_thermal_image_from_s3(device_id))
        rgb_image = load_rgb_image_from_s3(f'{device_id}/6_inch.png')
        gray_rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        is_hydra = device_type == self.HYDRA_DEVICE_NAME
        thermal_coordinates, _, _ = auto_point_detection.find_calibration_points_on_heatmap(
            thermal_image, is_hydra=is_hydra)
        rgb_coordinates, _, _ = auto_point_detection.find_calibration_points_on_rgb_photo(
            gray_rgb_image)

        if self.validate_calibration_points(rgb_coordinates) and (overwrite or not os.path.isfile(rgb_coordinates_file_path)):
        # if False:
            debug_rgb_image = rgb_image.copy()
            for x, y in rgb_coordinates:
                cv2.circle(debug_rgb_image, (int(x), int(y)), 0, (0, 0, 255), 10)
            self.see_image(debug_rgb_image, device_id)
            calibration_success = input("rgb Ok?").lower()[0] == 'y'
            if calibration_success:
                write_numpy_to_s3(
                    f"{device_id}/rgb_{device_id}_9element_coord.npy", rgb_coordinates)
            # else:  # enable to not do both if one's bad
            #     return calibration_success
        # if True:
        if self.validate_calibration_points(thermal_coordinates) and (overwrite or not os.path.isfile(trml_coordinates_file_path)):
            debug_thermal_image = thermal_image.copy()
        
            if debug_thermal_image.ndim != 3:
                debug_thermal_image = cv2.cvtColor(
                    debug_thermal_image, cv2.COLOR_GRAY2RGB)
        
            for x, y in thermal_coordinates:
                cv2.circle(debug_thermal_image, (int(x), int(y)), 0, (0, 0, 255), 10)
        
            debug_image = np.concatenate(
                (debug_thermal_image), axis=1)
            self.see_image(debug_thermal_image, device_id)
            # return False
            calibration_success = input("trml Ok?").lower()[0] == 'y'
            if calibration_success:
                write_numpy_to_s3(
                    f"{device_id}/trml_{device_id}_9element_coord.npy", thermal_coordinates)
            else:
                return calibration_success
        else:
            mask_matrix = np.zeros(self.IMAGE_SIZE)
        if not (self.validate_calibration_points(thermal_coordinates) and self.validate_calibration_points(rgb_coordinates)):
            return calibration_success

        thermal_coordinates = convert_coordinates_to_numpy_array(
            thermal_coordinates)
        rgb_coordinates = convert_coordinates_to_numpy_array(rgb_coordinates)
        parallax_calibrator = ParalaxCalibrator()
        mask_matrix, coordinate_map, sensitivity_matrix = parallax_calibrator(
            thermal_image,
            rgb_image,
            thermal_coordinates,
            rgb_coordinates
        )
        if self.validate_mask_matrix(mask_matrix):
            directory = f"{device_id}/calculated_transformations/{device_id}"
            calibration_success = True
            if not debug_mode:
                write_numpy_to_s3(
                    f"{directory}/mapped_coordinates_matrix_{device_type}_{device_idx}.npy", coordinate_map)
                write_numpy_to_s3(
                    f"{directory}/mapped_mask_matrix_{device_type}_{device_idx}.npy", mask_matrix)
                write_numpy_to_s3(
                    f"{directory}/sensitivity_correction_matrix_{device_type}_{device_idx}.npy", sensitivity_matrix)
                write_numpy_to_s3(
                    f"{device_id}/trml_{device_idx}_9element_coord.npy", thermal_coordinates)
                write_numpy_to_s3(
                    f"{device_id}/rgb_{device_idx}_9element_coord.npy", rgb_coordinates)
        debug_image = generate_debug_image(
            thermal_image, thermal_coordinates, rgb_image, rgb_coordinates, mask_matrix)

        if debug_mode and calibration_success:
            write_image_to_s3(f"{device_id}/debug_image.png", debug_image)
            update_data_json_on_s3(device_id, [("auto_cal", calibration_success)])
            write_numpy_to_s3(
                f"{directory}/mapped_coordinates_matrix_{device_type}_{device_idx}.npy", coordinate_map)
            write_numpy_to_s3(
                f"{directory}/mapped_mask_matrix_{device_type}_{device_idx}.npy", mask_matrix)
            write_numpy_to_s3(
                f"{directory}/sensitivity_correction_matrix_{device_type}_{device_idx}.npy", sensitivity_matrix)
            write_numpy_to_s3(
                f"{device_id}/trml_{device_idx}_9element_coord.npy", thermal_coordinates)
            write_numpy_to_s3(
                f"{device_id}/rgb_{device_idx}_9element_coord.npy", rgb_coordinates)
        else:
            write_image_to_s3(f"{device_id}/debug_image.png", debug_image)
            update_data_json_on_s3(device_id, [("auto_cal", calibration_success)])
        if not calibration_success:
            self.errors.append(device_id)
        return calibration_success


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    doc_path = '/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt'
    mp = Mapper()
    with open(doc_path) as csv_file:
        # reader = csv.reader(csv_file, delimiter='\t')
        reader = csv_file.read()  # allows tabs and spaces
        deviceList = []
        for line in reader.split("\n"):
            deviceList.append(line.split())
        for row in deviceList:
            print(f'---{"---".join(row)}----')
            success = mp.do_automatic_rgb_calibration_mapping(row[0])
            print(success)
        print(mp.errors)
        print("Failure rate:", len(mp.errors)/len(deviceList))
