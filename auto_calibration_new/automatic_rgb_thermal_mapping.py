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


    def do_automatic_rgb_calibration_mapping(self, device_id, debug_mode=False, overwrite=True):
        """Do automatic calibration mapping and send results to s3"""
        basePath = os.path.join("/home/canyon/S3bucket/", device_id)
        rgb_img_path = os.path.join(basePath, "6_inch.png")
        trml_img_path = os.path.join(basePath, "6_inch.npy")
        folder_path = basePath
        outputDir = os.path.join(basePath, 'calculated_transformations2')
        rgb_coordinates_file_path = folder_path + '/rgb_' + device_id + '_9element_coord.npy'
        trml_coordinates_file_path = folder_path + '/trml_' + device_id + '_9element_coord.npy'
        coordinatFiles = os.path.isfile(rgb_coordinates_file_path) and os.path.isfile(trml_coordinates_file_path)
        device_type, device_idx = get_device_type_and_idx(device_id)
        mask_exists = os.path.isfile(f"{device_id}/mapped_mask_matrix_{device_type}_{device_id}.npy")
        if mask_exists and not overwrite:
            print("exists")
            return "already exists"
        calibration_success = False
        thermal_image = load_thermal_image_from_s3(device_id)
        rgb_image = load_rgb_image_from_s3(f'{device_id}/6_inch.png')
        gray_rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        is_hydra = device_type == self.HYDRA_DEVICE_NAME
        thermal_coordinates, _, _ = auto_point_detection.find_calibration_points_on_heatmap(
            thermal_image, is_hydra=is_hydra)
        rgb_coordinates, _, _ = auto_point_detection.find_calibration_points_on_rgb_photo(
            gray_rgb_image)

        logging.info(
            f"Calibrating device {device_id}, type {device_type}, IDX {device_idx}")
        if self.validate_calibration_points(thermal_coordinates) and self.validate_calibration_points(rgb_coordinates):
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
                directory = f"{device_id}/calculated_transformations/{device_idx}"

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
                calibration_success = True
        elif self.validate_calibration_points(rgb_coordinates):
            debug_rgb_image = rgb_image.copy()
            for x, y in rgb_coordinates:
                cv2.circle(debug_rgb_image, (int(x), int(y)), 0, (0, 0, 255), 10)
            self.see_image(debug_rgb_image, device_id)
            calibration_success = input("Ok?").lower()[0] == 'y'
            if calibration_success:
                write_numpy_to_s3(
                    f"{device_id}/rgb_{device_idx}_9element_coord.npy", rgb_coordinates)
        elif self.validate_calibration_points(thermal_coordinates):
            debug_thermal_image = thermal_image.copy()
        
            if debug_thermal_image.ndim != 3:
                debug_thermal_image = cv2.cvtColor(
                    debug_thermal_image, cv2.COLOR_GRAY2RGB)
        
            for x, y in thermal_coordinates:
                cv2.circle(debug_thermal_image, (int(x), int(y)), 0, (0, 0, 255), 10)
        
            debug_image = np.concatenate(
                (debug_thermal_image), axis=1)
            self.see_image(debug_thermal_image, device_id)
            calibration_success = input("Ok?").lower()[0] == 'y'
            if calibration_success:
                write_numpy_to_s3(
                    f"{device_id}/trml_{device_idx}_9element_coord.npy", thermal_coordinates)
        else:
            mask_matrix = np.zeros(self.IMAGE_SIZE)

        debug_image = generate_debug_image(
            thermal_image, thermal_coordinates, rgb_image, rgb_coordinates, mask_matrix)

        if debug_mode and calibration_success:
            self.see_image(debug_image, device_id)
            calibration_success = input("Ok?").lower()[0] == 'y'
            if calibration_success:
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
            print(row)
            success = mp.do_automatic_rgb_calibration_mapping(row[0], debug_mode=True, overwrite=False)
            print(success)
        print(mp.errors)
        print("Failure rate:", len(mp.errors)/len(deviceList))
