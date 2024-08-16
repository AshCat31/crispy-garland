# Center-Vec

import json
import os
import statistics

import boto3
import botocore
import cv2
import numpy as np
import pandas as pd


def checkPath(file_path):
    result = s3client.list_objects(Bucket=bucket_name, Prefix=file_path)
    exists = False
    if "Contents" in result:
        exists = True
    return exists


# globals
count = count2 = 0
rgb_cen = (220, 260)
mag_list = []
deg_list = []
vec_list = []
sub_folder_list = []
error_list = []

# Setup boto3
cred = boto3.Session().get_credentials()
ACCESS_KEY = cred.access_key
SECRET_KEY = cred.secret_key
SESSION_TOKEN = cred.token
s3client = boto3.client(
    "s3",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    aws_session_token=SESSION_TOKEN,
)
bucket_name = "kcam-calibration-data"
# Create a paginator for listing objects in the bucket
# This is neccesary as S3 will force you to only get a maximum of 999 files at once and we need a lot more than that for this script
paginator = s3client.get_paginator("list_objects_v2")
# prefix='E661'
# Part_num='400-0102-03' # r3 heads
# dev_type='_hydra' # r3 heads
prefix = "100"
Part_num = "400-0103-02"  # hubs
dev_type = "_mosaic"  # hubs
local_directory = "vector_shifts/"

# Define the parameters for the list_objects_v2 call
list_objects_params = {
    "Bucket": bucket_name,
    "Prefix": prefix,
    "Delimiter": "/",
    "MaxKeys": 1000,  # Adjust the MaxKeys parameter as needed
}

# folder_paginator = s3client.get_paginator('list_objects_v2')

ROI_list = [
    "10000000c2094a7b",
    "10000000391f5111",
    "10000000dac6b1c8",
    "10000000a0fc1f63",
    "10000000cd020e5a",
    "10000000a91424c7",
    "10000000eef23bce",
    "10000000b0c862d4",
    "10000000e58e1c34",
    "1000000085d255ef",
    "1000000009456e4f",
    "E661AC8863514B24",
    "E661385283965A32",
    "E661AC8863846F27",
    "E66138528337BD2D",
    "E6613852837E4432",
    "E661AC8863253A24",
    "E661AC88637F3827",
    "E6613852832E5928",
    "E661385283391025",
    "E66138528387AB32",
    "E66138528349A732",
    "E661385283764732",
    "E661AC88631D9125",
    "E661385283675D28",
    "E6613852836C6D28",
    "E661AC8863973A24",
    "E661385283695932",
]
RMA_list = [
    "E661AC8863514B24",
    "E661385283965A32",
    "E661AC8863846F27",
    "E661AC88637DA821",
    "E66138528337BD2D",
    "E66138528382A92C",
    "E661385283302332",
    "E6613852837E4432",
    "E661AC8863253A24",
    "E661AC88637F3827",
    "E661385283453532",
    "E6613852834F3532",
    "E661385283674132",
    "E661385283652332",
    "E66138528357BF35",
    "E6613852838B3832",
    "E6613852831E6828",
    "E6613852831E4B28",
    "E66138528352B02D",
    "E661385283277D32",
    "E6613852832E5928",
    "E6613852833D6632",
    "E661385283391025",
    "E66138528380552C",
    "E66138528387AB32",
    "E66138528349A732",
    "E66138528345272C",
    "E661385283764732",
    "E661AC88631D9125",
    "E661AC8863117A25",
    "E661AC886349B227",
    "E661385283801D32",
    "E661AC88633A8427",
    "E661385283675D28",
    "E6613852836C6D28",
    "E661AC88631B8724",
    "E6613852836A4128",
    "E661AC8863973A24",
    "E661385283616832",
    "E6613852836AA42D",
    "E661385283695932",
    "E66138528348552C",
    "E6613852831E6A32",
    "E6613852836E4532",
    "E6613852837B5228",
    "E660D051136A5029",
    "E66138528362B12D",
    "E66138528332A92D",
    "E661385283947532",
    "E661385283367628",
    "E661AC88631B8724",
    "E66138528349A732",
    "E661385283675D28",
    "E661385283695932",
    "E661385283675D28",
    "E66138528387AB32",
    "E661385283783931",
    "E661AC88636B7824",
    "E661AC8863184A25",
    "E6613852833D7532",
    "E661AC8863087E25",
    "E661AC88639B7425",
    "E661385283650A32",
    "E661385283852732",
    "E661801017196335",
    "E66180101741BA35",
    "E6613852831A5732",
    "E661AC8863696727",
    "E661AC88633D24",
    "E661801017400D35",
    "E661801017244835",
    "E661801017686635",
    "E661801017625535",
    "E661801017427135",
    "E661801017565735",
    "E661801017913035",
    "E6618010170D5D35",
    "E6618010170C5E35",
    "E6613852835A3432",
    "E661801017817735",
    "E661801017353135",
    "E6618010175D1635",
    "E661801017518D35",
    "E661801017279F35",
    "E661801017995035",
    "E661801017289535",
    "E6618010172C7D35",
    "E660D051133C5028",
    "E661385283601131",
    "E661AC886330A022",
    "E661385283997628",
    "E661385283365328",
    "E66138528323892C",
    "E6613852837C5528",
    "E6613852839E5C32",
    "E66138528367B032",
    "E66138528362BF2D",
    "E661385283155A2C",
    "E66138528334372C",
    "E6613852838B3832",
    "E66138528357BF35",
    "E661385283674132",
    "E6613852837E4432",
    "E66138528337BD2D",
    "E661AC88637DA821",
    "E661801017399935",
    "E661385283556632",
    "E661385283544228",
    "E6618010177DA535",
    "E661385283482D32",
    "E661801017455D35",
    "E661801017737B35",
    "E661385283443A32",
    "E661385283389F35",
    "E661385283644B28",
    "E6613852831A4728",
    "E661385283335F32",
    "E66180101734AA35",
    "E6618010176FA335",
    "E661AC8863253A24",
    "E661385283965A32",
    "E661385283302332",
    "E661AC8863846F27",
    "E661AC8863514B24",
    "E66138528337BD2D",
    "E6613852837E4432",
]


def is_integrated(dev_id):
    found_good = False
    found_bad = False
    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=dev_id)
    for response in response_iterator:
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith(".jpeg"):
                if not key.endswith("3_rgb.jpeg"):
                    found_good = True
                else:
                    found_bad = True
    return found_good and not found_bad


def get_dir(device_id):
    parent = f"{device_id}/calculated_transformations/"
    response = s3client.list_objects_v2(Bucket=bucket_name, Prefix=parent)
    if "Contents" in response:
        for obj in response["Contents"]:
            # Extract directory name
            # Example: If obj['Key'] is 'device123/calculated_transformations/dir_name/'
            # Then dir_name would be 'dir_name'
            directory = obj["Key"].split("/")[2]
            return directory


def get_json(id):
    s3dir = "/home/canyon/S3bucket"
    key = f"{id}/data.json"
    try:
        with open(os.path.join(s3dir, key), "r") as json_file:
            json_content = json.load(json_file)
    except FileNotFoundError:
        try:
            s3client.download_file(
                Bucket=bucket_name, Key=key, Filename=os.path.join(s3dir, key)
            )
        except FileNotFoundError:
            return None
    with open(os.path.join(s3dir, key), "r") as json_file:
        json_content = json.load(json_file)
    # print(json_content)
    return json_content


# Iterate over pages of results using the paginator
for page in paginator.paginate(**list_objects_params):
    # Extract common prefixes (subfolders)
    for o in page.get("CommonPrefixes", []):
        sub_folder_list.append(o.get("Prefix"))

# files = os.listdir(local_directory)
# sub_folder_list = [fn[-20:-4]+"/" for fn in files]

for fn in sub_folder_list:
    if fn.startswith(prefix):
        print(fn)
    else:
        continue
    device_id = fn[:-1]
    path = f"{fn}data.json"
    if checkPath(path) and is_integrated(device_id):
        count += 1
        # print(count)
        try:
            bad_roi = device_id in ROI_list
            if device_id in RMA_list and not bad_roi:
                continue
            # print(bad_roi, device_id)
            # json_response = s3client.get_object(Bucket=bucket_name, Key=f'{device_id}/data.json')
            # json_file_content = json_response['Body'].read().decode('utf-8')
            # data_content = json.loads(json_file_content)
            json_file_content = get_json(device_id)
            if json_file_content is None:
                json_response = s3client.get_object(
                    Bucket=bucket_name, Key=f"{device_id}/data.json"
                )
                json_file_content = json_response["Body"].read().decode("utf-8")
                json_file_content = json.loads(json_file_content)
            if Part_num in json_file_content.values():
                dir_name = get_dir(device_id)
                mask_path = f"{device_id}/calculated_transformations/{dir_name}/mapped_mask_matrix{dev_type}_{dir_name}.npy"
                mask_path2 = f"{device_id}/calculated_transformations2/{dir_name}/mapped_mask_matrix{dev_type}_{dir_name}.npy"
                if checkPath(mask_path):  # Save the file to this path
                    pass
                elif checkPath(mask_path2):  # Save the file to this path
                    mask_path = mask_path2
                else:
                    continue
                # get mask/contour
                try:
                    mask = np.load(
                        f"{local_directory}mapped_mask_matrix{dev_type}_{device_id}.npy"
                    )
                except FileNotFoundError:
                    try:
                        s3client.download_file(
                            Bucket=bucket_name,
                            Key=mask_path,
                            Filename=os.path.join(
                                local_directory,
                                f"mapped_mask_matrix{dev_type}_{device_id}.npy",
                            ),
                        )
                    except FileNotFoundError:
                        s3client.download_file(
                            Bucket=bucket_name,
                            Key=mask_path2,
                            Filename=os.path.join(
                                local_directory,
                                f"mapped_mask_matrix{dev_type}_{device_id}.npy",
                            ),
                        )
                mask = np.load(
                    f"{local_directory}mapped_mask_matrix{dev_type}_{device_id}.npy"
                )
                mask = mask.astype(np.uint8) * 255
                mask_edges = cv2.Canny(mask, 30, 200)
                mask_edges_contours, _ = cv2.findContours(
                    mask_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                outermost_contour = mask_edges_contours[0]  # buidling the contour
                # axs[port].plot(statistics.mean(np.nonzero(mask_map)[1]), statistics.mean(np.nonzero(mask_map)[0]),'o', markersize=10, color='magenta', zorder=8888)
                # get centroid
                # M = cv2.moments(outermost_contour)
                # if M["m00"] != 0:
                #     therm_y = int(M["m10"] / M["m00"])
                #     therm_x = int(M["m01"] / M["m00"])
                # else:
                #     therm_y, therm_x = 0, 0
                # therm_cen=(therm_x,therm_y)
                # print(therm_cen)
                therm_x = statistics.mean(np.nonzero(mask)[0])
                therm_y = statistics.mean(np.nonzero(mask)[1])
                therm_cen = (therm_x, therm_y)
                # print(therm_cen)
                # get vector
                dX = rgb_cen[0] - therm_cen[0]
                dY = rgb_cen[1] - therm_cen[1]
                dif_mag = np.sqrt(dX**2 + dY**2)
                angle_rad = np.arctan2(
                    dY, dX
                )  # atan2(dY, dX) gives the angle in the standard coordinate system
                angle_deg = np.degrees(angle_rad)
                vec = (dif_mag, angle_deg)
                mag_list.append(dif_mag)
                deg_list.append(angle_deg)
                vec_list.append((*vec, device_id, bad_roi))
                # print(vec_list)
                # increment successful r3 head count
                count2 += 1
                print(f"C2:{count2}")
        except botocore.errorfactory.ClientError as e:
            error_list.append(device_id)

print("done")
# plt.savefig('Vec_test.png')
# plt.show()
df = pd.DataFrame(vec_list)
df.to_csv("Test_vec_integrated_hubs.csv", index=False)
print(
    "error list:", error_list
)  # List of Heads in S3 that did not have 1 or more of the required files.
# A fix for this exists in the S3 looping file and the QA check scripts but it was not implmented here as I moved on to other projects.
