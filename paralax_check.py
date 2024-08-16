__author__ = "Delta Thermal Inc."
__copyright__ = """
    Copyright 2018-2023 Delta Thermal Inc.

    All Rights Reserved.
    Covered by one or more of the Following US Patent Nos. 10,991,217,
    Other Patents Pending.
"""

import io
import json

import boto3
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from s3_setup import S3Setup


def main():
    # Setup device list
    device_list = []
    doc_path = "/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt"
    with open(doc_path, "r") as file:  # allow not just tabs as delimiters
        for line in file:
            values = line.split()
            device_list.append(values[0])

    # Setup boto3

    s3c = S3Setup()
    s3client, bucket_name = s3c()

    hub_base_image = [
        "/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_one.jpeg",
        "/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_two.jpeg",
        "/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_three.jpeg",
        "/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_four.jpeg",
        "/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_five.jpeg",
        "/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_six.jpeg",
    ]

    head_base_image = [
        "/home/canyon/Test_Equipment/head_alignment_test/port0_one.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port0_two.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port0_three.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port0_four.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port0_five.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port0_six.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port1_one.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port1_two.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port1_three.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port1_four.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port1_five.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port1_six.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port2_one.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port2_two.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port2_three.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port2_four.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port2_five.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port2_six.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port3_one.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port3_two.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port3_three.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port3_four.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port3_five.jpeg",
        "/home/canyon/Test_Equipment/head_alignment_test/port3_six.jpeg",
    ]

    for _device_id in device_list:
        parallax_check(
            _device_id, s3client, _bucket_name, hub_base_image, head_base_image
        )


def parallax_check(_device_id, s3client, _bucket_name, hub_base_image, head_base_image):
    print(_device_id)
    try:
        json_response = s3client.get_object(
            Bucket=_bucket_name, Key=f"{_device_id}/data.json"
        )
    except:
        print(f"{_device_id}/JSON does not exist")
        return
    json_file_content = json_response["Body"].read().decode("utf-8")
    data_content = json.loads(json_file_content)
    camera_id = data_content["camera_id"]

    try:
        if _device_id.startswith("E661"):
            key = f"{_device_id}/calculated_transformations2/{_device_id}/mapped_mask_matrix_hydra_{_device_id}.npy"
            mask_response = s3client.get_object(Bucket=_bucket_name, Key=key)
            # mask_response = s3client.get_object(Bucket=_bucket_name, Key=f'{_device_id}/calculated_transformations/{camera_id}/mapped_mask_matrix_mosaic_{camera_id}.npy')
            _base_image = head_base_image
            x, y = 4, 6
        else:
            key = f"{_device_id}/calculated_transformations2/{_device_id}/mapped_mask_matrix_mosaic_{_device_id}.npy"
            mask_response = s3client.get_object(Bucket=_bucket_name, Key=key)
            # mask_response = s3client.get_object(Bucket=_bucket_name, Key=f'{_device_id}/calculated_transformations/{_device_id}/mapped_mask_matrix_hydra_{_device_id}.npy')
            _base_image = hub_base_image
            x, y = 2, 3
    except Exception as e:
        print(e, "Did you putCal?")
        return
    mask_bytes = io.BytesIO(mask_response["Body"].read())
    mask_bytes.seek(0)

    mask_map = np.load(mask_bytes)
    mask_map = mask_map[100:420, 100:340]
    mask_map = mask_map.astype(np.uint8) * 255

    mask_edges = cv2.Canny(mask_map, 30, 200)
    mask_edges_contours, _ = cv2.findContours(
        mask_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    fig, axs = plt.subplots(x, y)
    fig.suptitle(_device_id)
    for val in range(len(_base_image)):
        rgb_img = mpimg.imread(_base_image[val])
        if len(rgb_img.shape) == 3:
            r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
            rgb_img = 0.2989 * r + 0.5870 * g + 0.1140 * b

        cv2.drawContours(rgb_img, mask_edges_contours, -1, (255, 255, 255), 1)
        row, col = divmod(val, max(x, y))
        axs[row][col].imshow(rgb_img, cmap="gray")
        axs[row][col].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0.01, bottom=0, top=0.95)
    plt.show()


if __name__ == "__main__":
    main()
