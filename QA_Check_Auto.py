# This is a copy of the version of this script found on Test_Equpiment Branch "QA_Check" as of 6/6/24
# This is the updated QA Check script that accuratly Checks for what % of an ROI is outside the mask border.
# This script is designed for Internal QA Checking, and has the ability to show an image of ROI's that are outside the mask border for review.
# most recent update adds full support for hub devices


import json
import logging
import os
from io import BytesIO

import boto3
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def main():
    global logger
    logger = logging.getLogger(__name__)
    log_format = '%(levelname)-6s: %(message)s'
    logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.basicConfig(level=logging.WARN, format=log_format)

    # Setup boto3
    cred = boto3.Session().get_credentials()
    ACCESS_KEY = cred.access_key
    SECRET_KEY = cred.secret_key
    SESSION_TOKEN = cred.token
    global s3client
    s3client = boto3.client('s3',
                            aws_access_key_id=ACCESS_KEY,
                            aws_secret_access_key=SECRET_KEY,
                            aws_session_token=SESSION_TOKEN,
                            )
    #######################IMPORTNAT READ BEFORE RUNNING FILE############################
    local_directory = 'calibration'  # Change this to the relative path you want to download the mask map the roi
    # matrix and the jpeg image too on your machine. these files are deleted by the script after they are used so this
    # is just a temporary location for checking the ROI's of a head/hub

    # defs
    global _bucket_name
    _bucket_name = 'kcam-calibration-data'
    with open('QA_ids.txt', 'r') as file:
        for line in file:
            values = line.split()
            # Inputs
            device_id = values[0]
            try:
                serial_number = values[1]
            except IndexError:
                logger.warning("SN not found.")
                serial_number = "none"
            print(device_id + "   " + serial_number)
            view_image = 'N'
            is_outside = False
            outside_points = []
            string_index = 17
            review_input = None

            # output json bools defaults
            id_check = True
            serial_number_check = True
            Auto_ROI = False
            mask_matrix = False
            _json = False
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
            _jpeg = False

            device_type = get_device_type(device_id)
            # folder Id is used for file names/paths in heads and hubs (was device id for the heads only QA check)
            folder_id = device_id  # default value for heads
            # Defs for file paths in S3
            folder_path = f'{device_id}/'
            json_path = f'{device_id}/data.json'
            six_inch_png_path = f'{device_id}/6_inch.png'
            six_inch_npy_path = f'{device_id}/6_inch.npy'
            # the rest of the paths are defined after the data.json check section
            # Inputs
            # device_id = input("Enter Device ID: ")
            # serial_number = input("Enter serial Number: ")
            # Checks:
            if checkPath(folder_path):  # Checking that the folder exists
                folder = True
                if checkPath(json_path):  # checking that the data.json exists
                    _json = True
                    try:
                        json_response = s3client.get_object(Bucket=_bucket_name, Key=f'{device_id}/data.json')
                        json_file_content = json_response['Body'].read().decode('utf-8')  # downloading the json
                        data_content = json.loads(json_file_content)
                        # saving the data from the json
                        camera_id = data_content['camera_id']
                        if device_type == 'mosaic':  # the most common missing value is SN so we set the folder id as early as posable before the try except gets triggered
                            folder_id = camera_id
                        js_device_id = data_content['device_id']
                        js_device_type = data_content['device_type']
                        hostname = data_content['hostname']
                        hardware_id = data_content['hardware_id']
                        qr_code = data_content['qr_code']
                        part_number = data_content['part_number']
                        js_serial_number = data_content['serial_number']

                        # Checking if inputs match data.json
                        if device_id != js_device_id:
                            logger.error(" Device Id in Data.json is not the same as input.")
                            id_check = False
                        if serial_number != js_serial_number:
                            serial_number_check = False
                            if serial_number != 'none':
                                logger.error(" Serial Number in Data.json is not the same as input.")
                            print("\tSN is", js_serial_number)
                        # printing the json data for review by use
                        logger.info("Data from json File: ")
                        logger.info("Device ID:" + device_id)
                        logger.info("Device Type:" + device_type)
                        logger.info("Hostname:" + hostname)
                        logger.info("Hardware ID:" + hardware_id)
                        logger.info("Camera ID:" + camera_id)
                        logger.info("QR Code:" + qr_code)
                        logger.info("Part Number:" + part_number)
                        logger.info("Serial Number:" + js_serial_number)
                    except KeyError:
                        logger.error("Json File Incomplete")
                        _json = False
                        serial_number_check = False  # if the json was not complete odds are the SN check if false
                        # the user should review the data.json for a incomplete error anyway though

                else:
                    logger.error(" data.json does not extist in S3")
                    _json = False
                # Remander of the path definitions, needed here as device type and folder id will not be correctly defined for hub devices
                nine_points_rgb_path = f'{device_id}/rgb_{folder_id}_9element_coord.npy'
                nine_points_trml_path = f'{device_id}/trml_{folder_id}_9element_coord.npy'
                # Defs for calculated transforms Paths in S3
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
                if checkPath(six_inch_png_path):
                    six_inch_png = True
                else:
                    logger.error(" 6inch.png does not exist in S3")

                if checkPath(six_inch_npy_path):
                    six_inch_npy = True
                else:
                    logger.error(" 6inch.npy does not exist in S3")

                if checkPath(nine_points_rgb_path):
                    nine_element_RGB = True
                else:
                    # logger.error(" 9 Element Coord RGB does not exist in S3")
                    pass
                if checkPath(nine_points_trml_path):
                    nine_element_trml = True
                else:
                    # logger.error(" 9 Element Coord Trml does not exist in S3")
                    pass
                # checking for the cal files in the calculated transforms folder
                if checkPath(calculated_transforms_path):
                    calc_trans = True
                    if checkPath(Coord_path):
                        coord = True
                    else:
                        logger.error(" Mapped Coord Matrix does not exist in S3")

                    if checkPath(Mask_path):
                        mask_matrix = True
                    else:
                        logger.error(" Mapped Mask Matrix does not exist in S3")

                    if checkPath(Sense_path):
                        sense = True
                    else:
                        logger.error(" Sensitivity Correction Matrix does not exist in S3")

                    # Checking if the roi matrix exists so it can be used to check if we have roi's outside the thermal cam fov
                    if checkPath(ROI_path):
                        ROI_matrix = True
                        # File 1 roi
                        s3client.download_file(Bucket=_bucket_name,
                                               Key=f'{device_id}/calculated_transformations/{folder_id}/regions_of_interest_matrix_{device_type}_{folder_id}.npy',
                                               Filename=os.path.join(local_directory,
                                                                     f'regions_of_interest_matrix_{device_type}_{folder_id}.npy'))
                        # FIle 2 mask
                        s3client.download_file(Bucket=_bucket_name,
                                               Key=f'{device_id}/calculated_transformations/{folder_id}/mapped_mask_matrix_{device_type}_{folder_id}.npy',
                                               Filename=os.path.join(local_directory,
                                                                     f'mapped_mask_matrix_{device_type}_{folder_id}.npy'))

                        mask = np.load(
                            os.path.join(local_directory, f'mapped_mask_matrix_{device_type}_{folder_id}.npy'))
                        roi_arr = np.load(
                            os.path.join(local_directory, f'regions_of_interest_matrix_{device_type}_{folder_id}.npy'))
                        roi_map = roi_arr
                        # mask setup
                        mask = mask.astype(np.uint8) * 255
                        mask_edges = cv2.Canny(mask, 30, 200)
                        mask_edges_contours, _ = cv2.findContours(mask_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        outermost_contour = mask_edges_contours[0]  # building the contour
                        # Jake's code to check if at least 50% of the points for a ROI are within the mask map countor that signifies the therm FOV
                        pass_fail = True
                        for roi in roi_map:
                            roi_x_loc_rgb = np.uint16(roi[:, :, 0])
                            roi_y_loc_rgb = np.uint16(roi[:, :, 1])

                            masked_roi_check = mask[roi_y_loc_rgb, roi_x_loc_rgb]

                            if np.count_nonzero(masked_roi_check) < 50:
                                pass_fail = False
                        if pass_fail == False:  # for now if any roi xy points are outside of the mask contour the user is then asked to review the image.
                            # Manual review
                            Auto_ROI = False
                            man_rev_req = True
                            logger.error("Auto ROI check FAILED.")
                            print("Do you wnat to start the manual review?")
                            view_image = input("Type 'Y' for yes or 'N' for No: ").upper()
                            if view_image == 'Y':
                                logger.info("Press 'Y' to close Image")
                                img_path = getIMG(local_directory, string_index, _bucket_name,
                                                  device_id)  # getting image

                                if img_path != "none":  # check to make sure there are jpeg files in S3
                                    _jpeg = True

                                    if manualReview(mask_edges_contours, roi_map, img_path,
                                                    device_id):  # if user hits Y
                                        logger.info("Manual Review Complete")
                                        man_rev = True
                                    else:  # if user hits N
                                        logger.info('Manual Complete')
                                        man_rev = True
                                else:  # If there are no jpeg files in S3
                                    logger.error(" No Jpeg files exist in S3. Cannot continue Manual Review")
                                    _jpeg = False
                                    man_rev = False
                            else:
                                print("Manual Review Declined")
                                man_rev = False
                        else:  # if auto ROI check passed
                            logger.info("All ROI's are in thermal cam FOV")
                            Auto_ROI = True
                            man_rev = False
                            man_rev_req = False
                            logger.info("All Cal Data Checks Complete")
                    else:  # if the checkPath(ROI) failed
                        # logger.error(" ROI Matrix does not exist in S3")
                        pass
                else:  # if the checkPath(calculated_transforms) failed
                    logger.info("Calculated transforms folder does not exist in S3")

            else:  # if the checkPath(device_id) failed
                logger.error(" This device has no files in S3")
                folder = False
            logger.info("All possible Cal Data Checks Complete")

            # Cleaning files used for ROI check out of the local directory
            if os.path.exists(f'{local_directory}/regions_of_interest_matrix_{device_type}_{folder_id}.npy'):
                os.remove(f'{local_directory}/regions_of_interest_matrix_{device_type}_{folder_id}.npy')
            if os.path.exists(f'{local_directory}/mapped_mask_matrix_{device_type}_{folder_id}.npy'):
                os.remove(f'{local_directory}/mapped_mask_matrix_{device_type}_{folder_id}.npy')
            if man_rev_req and img_path is not None:
                if os.path.exists(img_path):
                    os.remove(img_path)

            if folder:  # Can't put QA_Check.json in device folder of the folder does not exist

                if not man_rev:
                    if not Auto_ROI:
                        if _jpeg and ROI_matrix and mask_matrix:  # user hit N
                            man_rev_json = 'User Declined Review'
                        else:  # cant run review
                            man_rev_req = False
                            man_rev_json = 'Missing Prerequisites'
                    else:  # auto worked so man review was not needed
                        man_rev_json = "Auto ROI Check Passed"
                else:  # user hit Y
                    man_rev_json = "Manual Review Completed"

                # printing Results to QA_Check.json
                json_key = f'{device_id}/QA_Check_Dev.json'
                QA_Check = {
                    "Data.json Exists And is Complete": _json,
                    "Input Device Id Matches ID in json": id_check,
                    "Input Serial Number Matches SN in json": serial_number_check,
                    "6_inch.png Exists": six_inch_png,
                    "6_inch.npy Exists": six_inch_npy,
                    f"rgb_{folder_id}9elementcoord.npy Exists": nine_element_RGB,
                    f"trml_{folder_id}_9elementcoord.npy Exists": nine_element_trml,
                    "Calculated Transforms Folder Exists": calc_trans,
                    "Mapped Coordinates Matrix Exists": coord,
                    "Mapped Mask Matrix Exists": mask_matrix,
                    "Regions of Interest Matrix Exists": ROI_matrix,
                    "Sensitivity Correction Matrix Exists": sense,
                    "Auto ROI Check Pass": Auto_ROI,
                    "Manual Review required": man_rev_req,
                    "The ROIs were Reviewed by the user": man_rev,
                    "ROI Review Status": man_rev_json
                }
                json_data = json.dumps(QA_Check, indent=4)
                json_bytes = BytesIO(
                    json_data.encode('utf-8'))  # uploading json to S3 without saving the json to the hard drive
                s3client.put_object(Bucket=_bucket_name, Key=json_key, Body=json_bytes)

                logger.info("json file written to S3")


# Function that checks weather the device id belongs to a hub or a head
def get_device_type(device_id):
    device_id = device_id.strip()  # Remove any leading/trailing whitespace
    if device_id.startswith("10000"):
        return "mosaic"
    elif device_id.startswith("E66"):
        return "hydra"
    else:
        return "unknown"


# function for manual review's image interaction
def on_key_press(event):
    global review_input  # Reference the outer variable

    # Check if the key pressed is 'Y' or 'N'
    if event.key == 'Y':
        review_input = True
        plt.close()  # Close the plot after capturing the response
    elif event.key == 'N':
        review_input = False
        plt.close()  # Close the plot after capturing the response


# Function to check if a file exists in S3
def checkPath(file_path):
    result = s3client.list_objects(Bucket=_bucket_name,
                                   Prefix=file_path)  # This pulls all files in S3 with the 'File Path' Prefix
    exists = False
    # Because the paths Defined above are file paths and not folder paths if anything is found at that path it can only be the file, therefore it exists
    if 'Contents' in result:
        exists = True
    return exists


# This function finds the image in S3 and then downloads it
def getIMG(local_directory, string_index, _bucket_name, device_id):
    img_directory = local_directory
    # Because the jpeg images have a complicated file name that cannot be determined by data in S3 we have to find the file name before we can download it
    all_objects = s3client.list_objects(Bucket='kcam-calibration-data')
    response = s3client.list_objects_v2(
        Bucket=_bucket_name,
        Prefix=device_id)  # This grabs a list of every object in the 'device_id'S3 folder
    jpeg_files = []

    # Iterate over 'Contents' and check for files that end in JPEG
    for item in response['Contents']:
        key = item.get('Key', '')  # Get the 'Key' value
        if key.endswith('.jpeg'):  # Check if the key ends with .jpeg
            jpeg_files.append(key)  # Add to the list of JPEG files
    if len(jpeg_files) > 0:  # Checking that there are jpeg files to find
        img_file_name = jpeg_files[0][string_index:string_index + 50]
        # now that we have found the (first) jpeg file we can download it
        s3client.download_file(Bucket=_bucket_name,
                               Key=f'{jpeg_files[0]}',
                               Filename=os.path.join(local_directory, f'{img_file_name}'))
        logger.info("img downloaded")
        # img = mpimg.imread(f'{local_directory}/{img_file_name}')
        img_path = f'{local_directory}/{img_file_name}'

        return img_path
    else:
        return 'none'


# Function for the user to check if the roi that has points outside the thermal cam fov is passable or if the roi is too far outsied fov for even subpixel data
def manualReview(mask_edges_contours, roi, img_path, device_id):
    global review_input
    img = mpimg.imread(img_path)
    logger.info("img loaded")
    # padding the image
    if len(img.shape) == 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        img = 0.2989 * r + 0.5870 * g + 0.1140 * b
    img_padded = np.zeros((520, 440)).astype(np.uint8)
    img_padded[100:420, 100:340] = img
    img = img_padded

    # drawing the contours
    contour_img1 = img.copy()
    cv2.drawContours(contour_img1, mask_edges_contours, -1, (255, 255, 255), 1)

    fig, ax = plt.subplots(1, 1)
    # Set title 
    ax.set_title(f'ROI QA Image for {device_id}')

    # Display the image on the subplot
    ax.imshow(contour_img1, cmap='gray')

    # Plot the regions on the  subplot
    for region in roi:
        ax.plot(region[:, :, 0], region[:, :, 1], 'ro', markersize=2)
        # Connect the key press event to the on_key_press function
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    # Show the plot
    plt.show()

    return review_input


if __name__ == "__main__":
    main()
