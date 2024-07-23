# Copy of the version of this script found on Test_Equipment Branch "QA_Check" as of 6/6/24
# Updated QA Check script that accurately Checks for what % of an ROI is outside the mask border.
# Designed for Internal QA Checking, and can show the image, ROIs, and mask
# Most recent update adds full support for hub devices


import json
import logging
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np

from s3_setup import S3Setup


def configure_logger():
    logger = logging.getLogger(__name__)
    log_format = '%(levelname)-6s: %(message)s'
    logging.basicConfig(format=log_format)
    logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    return logger


class JSONChecker:
    def __init__(self, values, s3client, bucket_name, logger) -> None:
        self.logger = logger
        self._s3client = s3client
        self.bucket_name = bucket_name
        self.device_id = values[0]
        self.serial_number = self.get_serial_number(values)

        # output json bools defaults
        self.id_check = self.serial_number_check = True
        self.exists_json = self.exists_6in_png = self.exists_6in_npy = False
        self.exists_9ele_trml = self.exists_9ele_rgb = self.exists_calc_trans = False
        self.exists_coord = self.exists_sense = self.exists_mask_matrix = False
        self.exists_roi_matrix = self.man_rev_req = self.auto_check_pass = False
        self.valid_jpeg = self.man_rev = self.review_input = False

        self.device_type = self.get_device_type(self.device_id)
        self.inner_dir = self.device_id  # inner folder is deviceID for heads but cameraID for hubs
        self.json_path = f'{self.device_id}/data.json'
        self.six_inch_png_path = f'{self.device_id}/6_inch.png'
        self.six_inch_npy_path = f'{self.device_id}/6_inch.npy'

    def get_serial_number(self, values):
        if len(values) == 2:
            return values[1]
        else:
            self.logger.warning("SN not found.")
            return "none"

    def qa_device(self):
        if self.check_path(f'{self.device_id}/'):
            self.validate_json()
            self.exists_6in_png = self.log_exists(self.six_inch_png_path, "6inch.png")
            self.exists_6in_npy = self.log_exists(self.six_inch_npy_path, "6inch.npy")
            self.exists_9ele_rgb = self.log_exists(f'{self.device_id}/rgb_{self.inner_dir}_9element_coord.npy')
            self.exists_9ele_trml = self.log_exists(f'{self.device_id}/trml_{self.inner_dir}_9element_coord.npy')

            self.inner_dir_path = f'{self.device_id}/calculated_transformations/{self.inner_dir}'
            self.exists_calc_trans = self.check_path(self.inner_dir_path)
            if self.exists_calc_trans:
                self.check_inner_paths()
            else:  # if the self.check_path(calculated_transforms) failed
                self.logger.info("Calculated transforms folder does not exist in S3")

            self.get_status()
            self.save_json()
        else:  # if the self.check_path(self.device_id) failed
            self.logger.error("This device has no files in S3")

        self.logger.info("All possible Cal Data Checks Complete")

    def validate_json(self):
        self.exists_json = self.check_path(self.json_path)
        if self.exists_json:
            try:
                json_response = self._get_s3_object(f'{self.device_id}/data.json')
                data_content = json.loads(json_response['Body'].read().decode('utf-8'))
                self.read_data(data_content)
                self.log_info()

                self.id_check = self.check_js_matches(self.device_id, self.js_device_id, "Device ID")
                self.serial_number_check = self.check_js_matches(self.serial_number, self.js_serial_number, "Serial Number")
            except KeyError:
                self.logger.error("Json File Incomplete")
                print("\tKeys are", ', '.join(list(data_content.keys())))
                self.exists_json = False
        else:
            self.logger.error("data.json does not exist in S3")

    def _get_s3_objects(self, Prefix):
        return self._s3client.list_objects_v2(Bucket=self.bucket_name,Prefix=Prefix)

    def _get_s3_object(self, Key):
        return self._s3client.get_object(Bucket=self.bucket_name,Key=Key)

    def check_path(self, file_path):
        """Check if a file exists in S3"""
        return 'Contents' in self._get_s3_objects(file_path)  # if anything was found, it can only be the file

    def log_exists(self, path, file_name=''):
        exists = self.check_path(path)
        if not exists and file_name:
            self.logger.error(path + " does not exist in S3")
        return exists

    def check_js_matches(self, local, js, name):
        if local != js:
            self.logger.error(f"{name} in data.json is not the same as input.")
            print(f"\t{name} is", js, "not", local)
            return False
        return True

    def read_data(self, data_content):
        self.camera_id = data_content['camera_id']
        if self.device_type == 'mosaic':  # set the hub's folder id before any exceptions
            self.inner_dir = self.camera_id
        self.js_device_id = data_content['device_id']
        self.hostname = data_content['hostname']
        self.hardware_id = data_content['hardware_id']
        self.qr_code = data_content['qr_code']
        self.part_number = data_content['part_number']
        self.js_serial_number = data_content['serial_number']
        self.work_order = data_content['work_order']
        print(self.work_order)

    def get_device_type(self, device_id):
        """Determines whether a device is a head or hub"""
        return {"100": "mosaic", "E66": "hydra"}.get(device_id.strip()[:3], "unknown")
    
    def load_file_from_s3(self, key):
        try:
            raw_response = self._get_s3_object(Key=key)
            raw_bytes = BytesIO(raw_response["Body"].read())
            raw_bytes.seek(0)
            return raw_bytes
        except Exception as e:
            print(f"Error downloading from S3: {e}")
            return None

    def load_array_from_s3(self, key):
        """Downloads a .npy file from S3 and loads it into a NumPy array."""
        raw_bytes = self.load_file_from_s3(key)
        return np.load(raw_bytes) if raw_bytes else None
        
    def load_rgb_image_from_s3(self, key: str):
        """Download rgb image from s3"""
        raw_bytes = self.load_file_from_s3(key)
        if raw_bytes is not None:
            raw_image_bytes = np.frombuffer(raw_bytes.read(), dtype=np.uint8)
            rgb_image = cv2.imdecode(raw_image_bytes.astype(np.uint8), cv2.IMREAD_COLOR)
            if np.max(rgb_image) <= 1:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            return rgb_image
        return None

    def find_img(self):
        """finds the jpeg image in S3 and returns the path"""
        # jpeg images have a complicated file name that cannot be determined; have to find the file name
        response = self._get_s3_objects(self.device_id)
        for item in response['Contents']:  # Check for files that end in JPEG
            key = item.get('Key', '')
            if key.endswith('.jpeg'):
                return f'{key}'
        return None
    
    def check_inner_paths(self):
        coord_path = (f'{self.inner_dir_path}/mapped_coordinates_matrix_{self.device_type}_{self.inner_dir}.npy')
        sense_path = (f'{self.inner_dir_path}/sensitivity_correction_matrix_{self.device_type}_{self.inner_dir}.npy')
        roi_path = (f'{self.inner_dir_path}/regions_of_interest_matrix_{self.device_type}_{self.inner_dir}.npy')
        mask_path = (f'{self.inner_dir_path}/mapped_mask_matrix_{self.device_type}_{self.inner_dir}.npy')

        self.exists_coord = self.log_exists(coord_path, "Mapped Coord Matrix")
        self.exists_sense = self.log_exists(sense_path, "Sensitivity Correction Matrix")

        self.exists_mask_matrix = self.log_exists(mask_path, "Mapped Mask Matrix")
        self.exists_roi_matrix = self.check_path(roi_path)
        if self.exists_roi_matrix and self.exists_mask_matrix:
            self.verify_rois(roi_path, mask_path)
    
    def verify_rois(self, roi_path, mask_path):
        self.roi_map = self.load_array_from_s3(roi_path)
        self.mask = self.load_array_from_s3(mask_path).astype(np.uint8) * 255

        self.auto_check_pass = self.verify_rois_inside_contour()
        self.man_rev_req = not self.auto_check_pass
        if True:  # user can review images where 50%+ ROIs outside mask
        # if not self.auto_check_pass:  # user can review images where 50%+ ROIs outside mask
            self.logger.error("Auto ROI check FAILED.")
            self.start_review()
        else:  # if auto ROI check passed
            self.logger.info("Auto check passed; all ROIs are in thermal cam FOV")

    def start_review(self):
        print("Do you want to start the manual review?")
        if input("Type 'Y' for yes or 'N' for No:\n")[0].upper() == 'Y':
            self.img_path = self.find_img()

            self.valid_jpeg = self.img_path != None  # check to make sure there are jpeg files in S3
            if self.valid_jpeg:
                self.mask_edges_contours, _ = cv2.findContours(cv2.Canny(self.mask, 30, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                self.man_rev = self.manual_review()
                self.logger.info("Manual Review Complete")
            else:  # If there are no jpeg files in S3
                self.logger.error("No Jpeg files exist in S3. Cannot continue Manual Review")
                self.man_rev_req = False
        else:
            print("Manual Review Declined")

    def verify_rois_inside_contour(self):
        """Jake's code to check if 50%+ of the points for a ROI are within mask map countor"""    
        for roi in self.roi_map:
            roi_x_loc_rgb = np.uint16(roi[:, :, 0])
            roi_y_loc_rgb = np.uint16(roi[:, :, 1])
            masked_roi_check = self.mask[roi_y_loc_rgb, roi_x_loc_rgb]
            if np.count_nonzero(masked_roi_check) < 50:
                return False
        return True

    def on_key_press(self, event):
        """Handles manual review's image interaction"""
        if event.key.upper() in 'YN':
            self.review_input = event.key.upper() == 'Y'
            plt.close()  # Close the plot after capturing the response

    def pad_image(self, img):
        if len(img.shape) == 3:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = 0.2989 * r + 0.5870 * g + 0.1140 * b
        img_padded = np.zeros((520, 440)).astype(np.uint8)
        img_padded[100:420, 100:340] = img
        return img_padded

    def plot_img(self, ax):
        img = self.pad_image(self.load_rgb_image_from_s3(self.img_path))
        cv2.drawContours(img, self.mask_edges_contours, -1, (255, 255, 255), 1)
        for region in self.roi_map:
            ax.plot(region[:, :, 0], region[:, :, 1], 'ro', markersize=2)
        ax.imshow(img, cmap='grey')

    def manual_review(self):
        """Has user check if the roi is passable or too far outside fov for even subpixel data"""
        fig, ax = plt.subplots()
        ax.set_title(f'Y for passing, N for failing')
        self.plot_img(ax)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show()
        return self.review_input

    def get_status(self):
        if not self.man_rev:
            if not self.auto_check_pass:
                if self.valid_jpeg and self.exists_roi_matrix and self.exists_mask_matrix:  # user hit N
                    self.man_rev_status = 'User Declined Review'
                else:  # can't run review
                    self.man_rev_status = 'Missing Prerequisites'
            else:  # auto worked so man review was not needed
                self.man_rev_status = "Auto ROI Check Passed"
        else:  # user hit Y
            self.man_rev_status = "Manual Review Completed"

    def log_info(self):
        self.logger.info("Data from json File: ")
        self.logger.info("Device ID:" + self.device_id)
        self.logger.info("Device Type:" + self.device_type)
        self.logger.info("Hostname:" + str(self.hostname))
        self.logger.info("Hardware ID:" + self.hardware_id)
        self.logger.info("Camera ID:" + self.camera_id)
        self.logger.info("QR Code:" + self.qr_code)
        self.logger.info("Part Number:" + self.part_number)
        self.logger.info("Serial Number:" + self.js_serial_number)
        self.logger.info("Work Order:" + self.work_order)

    def save_json(self):
        json_key = f'{self.device_id}/QA_Check_Dev.json'
        QA_Check = {
            "Data.json Exists And is Complete": self.exists_json,
            "Input Device Id Matches ID in json": self.id_check,
            "Input Serial Number Matches SN in json": self.serial_number_check,
            "6_inch.png Exists": self.exists_6in_png,
            "6_inch.npy Exists": self.exists_6in_npy,
            f"rgb_{self.inner_dir}_9elementcoord.npy Exists": self.exists_9ele_rgb,
            f"trml_{self.inner_dir}_9elementcoord.npy Exists": self.exists_9ele_trml,
            "Calculated Transforms Folder Exists": self.exists_calc_trans,
            "Mapped Coordinates Matrix Exists": self.exists_coord,
            "Mapped Mask Matrix Exists": self.exists_mask_matrix,
            "Regions of Interest Matrix Exists": self.exists_roi_matrix,
            "Sensitivity Correction Matrix Exists": self.exists_sense,
            "Auto ROI Check Pass": self.auto_check_pass,
            "Manual Review required": self.man_rev_req,
            "ROIs Reviewed by user": self.man_rev,
            "ROI Review Status": self.man_rev_status
        }
        json_data = json.dumps(QA_Check, indent=4)
        self._s3client.put_object(Bucket=self.bucket_name, Key=json_key, Body=BytesIO(json_data.encode('utf-8')))
        self.logger.info("json file written to S3")

def main():
    logger = configure_logger()
    s3c = S3Setup()
    s3client, bucket_name = s3c()
    with open("/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt", 'r') as file:
        lines = [line for line in file]
    for line in lines:
        values = line.split()
        jc = JSONChecker(values, s3client, bucket_name, logger)
        jc.qa_device()

if __name__ == "__main__":
    main()
