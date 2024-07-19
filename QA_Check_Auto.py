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

from s3_setup import setup_s3


def configure_logger():
    logger = logging.getLogger(__name__)
    log_format = '%(levelname)-6s: %(message)s'
    logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.basicConfig(level=logging.WARN, format=log_format)
    return logger

def check_exists(path, file_name=''):
    exists = check_path(path)
    if not exists and file_name:
        logger.error(" " + path + " does not exist in S3")
    return exists

def load_array_from_s3(bucket_name, key):
    """
    Downloads a single array file from S3 and loads it into a NumPy array.

    Args:
    - bucket_name (str): Name of the S3 bucket.
    - key (str): Key of the object in the S3 bucket.

    Returns:
    - numpy.ndarray: Loaded array from the downloaded file.
    """
    try:
        raw_response = s3client.get_object(Bucket=bucket_name, Key=key)
        raw_bytes = BytesIO(raw_response["Body"].read())
        raw_bytes.seek(0)
        array = np.load(raw_bytes)
        return array
    except Exception as e:
        print(f"Error downloading and loading array from S3: {e}")
        return None

def get_device_type(device_id):
    """Checks whether a device is a head or hub"""
    try:
        return {"100": "mosaic","E66": "hydra"}[device_id.strip()[:3]]
    except KeyError:
        return "unknown"

def on_key_press(event):
    """Handles manual review's image interaction"""
    global review_input
    if event.key.upper() in 'YN':
        review_input = event.key.upper() == 'Y'
        plt.close()  # Close the plot after capturing the response

def check_path(file_path):
    """Check if a file exists in S3"""
    result = s3client.list_objects(Bucket=bucket_name,Prefix=file_path)
    if 'Contents' in result:  # if anything was found, it can only be the file
        return True
    return False

def load_rgb_image_from_s3(key: str):
    """Download rgb image from s3"""
    try:
        raw_response = s3client.get_object(Bucket=bucket_name, Key=key)
        raw_bytes = BytesIO(raw_response["Body"].read())
        raw_bytes.seek(0)
    except Exception as e:
        print(f"Error downloading and loading array from S3: {e}")
        return None
    raw_image_np_bytes = np.asarray(
        bytearray(raw_bytes.read()), dtype=np.uint8)
    rgb_image = cv2.imdecode(raw_image_np_bytes, cv2.IMREAD_COLOR)
    if np.max(rgb_image) <= 1:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    return rgb_image

def get_img( bucket_name, device_id):
    """finds the jpeg image in S3 and then downloads it"""
    # jpeg images have a complicated file name that cannot be determined; we have to find the file name
    response = s3client.list_objects_v2(Bucket=bucket_name,Prefix=device_id)
    for item in response['Contents']:  # Check for files that end in JPEG
        key = item.get('Key', '')
        if key.endswith('.jpeg'):
            return f'{key}'
    return None

def verify_rois_inside_contour(roi_map, mask):
    """Jake's code to check if 50%+ of the points for a ROI are within mask map countor"""    
    for roi in roi_map:
        roi_x_loc_rgb = np.uint16(roi[:, :, 0])
        roi_y_loc_rgb = np.uint16(roi[:, :, 1])
        masked_roi_check = mask[roi_y_loc_rgb, roi_x_loc_rgb]
        if np.count_nonzero(masked_roi_check) < 50:
            return False
    return True

def manual_review(mask_edges_contours, roi, img_path):
    """Has user check if the roi is passable or too far outside fov for even subpixel data"""
    global review_input
    review_input = False
    img = load_rgb_image_from_s3(img_path)

    # padding the image
    if len(img.shape) == 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        img = 0.2989 * r + 0.5870 * g + 0.1140 * b
    img_padded = np.zeros((520, 440)).astype(np.uint8)
    img_padded[100:420, 100:340] = img
    img = img_padded

    cv2.drawContours(img, mask_edges_contours, -1, (255, 255, 255), 1)

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Y for passing, N for failing')
    ax.imshow(img, cmap='gray')

    for region in roi:
        ax.plot(region[:, :, 0], region[:, :, 1], 'ro', markersize=2)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.show()

    return review_input

def main():
    global logger
    logger = configure_logger()

    # Setup boto3
    global s3client, bucket_name
    s3client, bucket_name = setup_s3()

    # defs
    lines = []
    bucket_name = 'kcam-calibration-data'
    with open("/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt", 'r') as file:
        for line in file:
            lines.append(line)
    for line in lines:
        values = line.split()
        device_id = values[0]
        try:
            serial_number = values[1]
        except IndexError:
            logger.warning("SN not found.")
            serial_number = "none"
        # print(device_id + "   " + serial_number)
        view_image = 'N'
        img_path = None

        # output json bools defaults
        id_check = True
        serial_number_check = True
        Auto_ROI = False
        mask_matrix = False
        valid_json = False
        six_inch_png = False
        six_inch_npy = False
        nine_element_RGB = False
        nine_element_trml = False
        calc_trans = False
        coord = False
        sense = False
        ROI_matrix = False
        man_rev_req = False
        man_rev = False
        valid_jpeg = False

        device_type = get_device_type(device_id)
        folder_id = device_id  # folder is deviceID for heads but cameraID for hubs
        # Defs for file paths in S3
        folder_path = f'{device_id}/'
        json_path = f'{device_id}/data.json'
        six_inch_png_path = f'{device_id}/6_inch.png'
        six_inch_npy_path = f'{device_id}/6_inch.npy'

        # Checks:
        folder = check_path(folder_path)
        if folder:
            valid_json = check_path(json_path)
            if valid_json:
                try:
                    json_response = s3client.get_object(Bucket=bucket_name, Key=f'{device_id}/data.json')
                    json_file_content = json_response['Body'].read().decode('utf-8')
                    data_content = json.loads(json_file_content)

                    camera_id = data_content['camera_id']
                    if device_type == 'mosaic':  # set the hub's folder id before exceptions
                        folder_id = camera_id
                    js_device_id = data_content['device_id']
                    hostname = data_content['hostname']
                    hardware_id = data_content['hardware_id']
                    qr_code = data_content['qr_code']
                    part_number = data_content['part_number']
                    js_serial_number = data_content['serial_number']
                    print(data_content['work_order'])

                    # Checking if inputs match data.json
                    if device_id != js_device_id:
                        logger.error(" Device Id in Data.json is not the same as input.")
                        id_check = False
                        print("id is", js_device_id, "not", device_id)
                    if serial_number != js_serial_number:
                        serial_number_check = False
                        if serial_number != 'none':
                            logger.error(" Serial Number in Data.json is not the same as input.")
                        print("\tSN is", js_serial_number, "not", serial_number)

                    # logging the json data for review by use
                    logger.info("Data from json File: ")
                    logger.info("Device ID:" + device_id)
                    logger.info("Device Type:" + device_type)
                    logger.info("Hostname:" + str(hostname))
                    logger.info("Hardware ID:" + hardware_id)
                    logger.info("Camera ID:" + camera_id)
                    logger.info("QR Code:" + qr_code)
                    logger.info("Part Number:" + part_number)
                    logger.info("Serial Number:" + js_serial_number)

                except KeyError:
                    logger.error("Json File Incomplete")
                    print("\tKeys are", ', '.join(list(data_content.keys())))
                    valid_json = False
            else:
                logger.error(" data.json does not extist in S3")
                valid_json = False
            # path definitions, needed here or folder id will be wrong for hubs
            nine_points_rgb_path = f'{device_id}/rgb_{folder_id}_9element_coord.npy'
            nine_points_trml_path = f'{device_id}/trml_{folder_id}_9element_coord.npy'
            # calculated transforms Paths in S3
            calculated_transforms_path = f'{device_id}/calculated_transformations/{folder_id}'
            ROI_path = (
                f'{device_id}/calculated_transformations/{folder_id}/regions_of_interest_matrix_{device_type}_{folder_id}.npy')
            Mask_path = (
                f'{device_id}/calculated_transformations/{folder_id}/mapped_mask_matrix_{device_type}_{folder_id}.npy')
            Coord_path = (
                f'{device_id}/calculated_transformations/{folder_id}/mapped_coordinates_matrix_{device_type}_{folder_id}.npy')
            Sense_path = (
                f'{device_id}/calculated_transformations/{folder_id}/sensitivity_correction_matrix_{device_type}_{folder_id}.npy')

            # Checking for the cal files outside the calculated transforms folder
            six_inch_png = check_exists(six_inch_png_path, "6inch.png")
            six_inch_npy = check_exists(six_inch_npy_path, "6inch.npy")
            nine_element_RGB = check_exists(nine_points_rgb_path)
            nine_element_trml = check_exists(nine_points_trml_path)

            calc_trans = check_path(calculated_transforms_path)
            if calc_trans:
                coord = check_exists(Coord_path, "Mapped Coord Matrix")
                mask_matrix = check_exists(Mask_path, "Mapped Mask Matrix")
                sense = check_exists(Sense_path, "Sensitivity Correction Matrix")

                ROI_matrix = check_path(ROI_path)
                if ROI_matrix:
                    roi_map = load_array_from_s3(bucket_name,
                                           f'{device_id}/calculated_transformations/{folder_id}/regions_of_interest_matrix_{device_type}_{folder_id}.npy')
                    mask = load_array_from_s3(bucket_name,
                                           f'{device_id}/calculated_transformations/{folder_id}/mapped_mask_matrix_{device_type}_{folder_id}.npy')
                    mask = mask.astype(np.uint8) * 255

                    pass_fail = verify_rois_inside_contour(roi_map, mask)
                    man_rev = False
                    if pass_fail:  # user can review images where 50%+ ROIs outside mask
                        # Manual review
                        Auto_ROI = False
                        man_rev_req = True
                        logger.error("Auto ROI check FAILED.")
                        print("Do you want to start the manual review?")
                        view_image = input("Type 'Y' for yes or 'N' for No: ").upper()[0]
                        if view_image == 'Y':
                            logger.info("Press 'Y' to close Image")
                            img_path = get_img(bucket_name, device_id)

                            valid_jpeg = img_path != None  # check to make sure there are jpeg files in S3
                            if valid_jpeg:
                                mask_edges = cv2.Canny(mask, 30, 200)
                                mask_edges_contours, _ = cv2.findContours(mask_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                man_rev = manual_review(mask_edges_contours, roi_map, img_path)
                                logger.info("Manual Review Complete")
                            else:  # If there are no jpeg files in S3
                                logger.error(" No Jpeg files exist in S3. Cannot continue Manual Review")
                        else:
                            print("Manual Review Declined")
                    else:  # if auto ROI check passed
                        logger.info("All ROI's are in thermal cam FOV")
                        Auto_ROI = True
                        man_rev_req = False
                        logger.info("All Cal Data Checks Complete")
            else:  # if the check_path(calculated_transforms) failed
                logger.info("Calculated transforms folder does not exist in S3")

            if not man_rev:
                if not Auto_ROI:
                    if valid_jpeg and ROI_matrix and mask_matrix:  # user hit N
                        man_rev_status = 'User Declined Review'
                    else:  # can't run review
                        man_rev_req = False
                        man_rev_status = 'Missing Prerequisites'
                else:  # auto worked so man review was not needed
                    man_rev_status = "Auto ROI Check Passed"
            else:  # user hit Y
                man_rev_status = "Manual Review Completed"

            # printing Results to QA_Check.json
            json_key = f'{device_id}/QA_Check_Dev.json'
            QA_Check = {
                "Data.json Exists And is Complete": valid_json,
                "Input Device Id Matches ID in json": id_check,
                "Input Serial Number Matches SN in json": serial_number_check,
                "6_inch.png Exists": six_inch_png,
                "6_inch.npy Exists": six_inch_npy,
                f"rgb_{folder_id}_9elementcoord.npy Exists": nine_element_RGB,
                f"trml_{folder_id}_9elementcoord.npy Exists": nine_element_trml,
                "Calculated Transforms Folder Exists": calc_trans,
                "Mapped Coordinates Matrix Exists": coord,
                "Mapped Mask Matrix Exists": mask_matrix,
                "Regions of Interest Matrix Exists": ROI_matrix,
                "Sensitivity Correction Matrix Exists": sense,
                "Auto ROI Check Pass": Auto_ROI,
                "Manual Review required": man_rev_req,
                "ROIs Reviewed by user": man_rev,
                "ROI Review Status": man_rev_status
            }
            json_data = json.dumps(QA_Check, indent=4)
            s3client.put_object(Bucket=bucket_name, Key=json_key, Body=BytesIO(json_data.encode('utf-8')))
            logger.info("json file written to S3")
        else:  # if the check_path(device_id) failed
            logger.error(" This device has no files in S3")
        logger.info("All possible Cal Data Checks Complete")


        # if folder:  # Can't put QA_Check.json in device folder if the folder does not exist

if __name__ == "__main__":
    main()
