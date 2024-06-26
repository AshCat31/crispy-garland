__author__ = 'Delta Thermal Inc.'
__copyright__ = """
    Copyright 2018-2023 Delta Thermal Inc.

    All Rights Reserved.
    Covered by one or more of the Following US Patent Nos. 10,991,217,
    Other Patents Pending.
"""

import io
import statistics

import boto3
import matplotlib.pyplot as plt
import numpy as np


def main():
    device_list = []
    doc_path = '/home/canyon/Test_Equipment/QA_ids.txt'
    with open(doc_path, 'r') as file:
        for line in file:
            values = line.split()
            device_list.append(values[0])

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
    bucket_name = 'kcam-calibration-data'

    device_type_dict = {"100": ("_mosaic",), "E66": ("_hydra",)}
    coverage_percents = []
    for i, device_id in enumerate(device_list):
        print(i + 1, device_id)
        device_type = device_type_dict[device_id[:3]][0]
        try:
            mask_response = get_mask("", device_id, device_type, bucket_name, coverage_percents)
        except Exception as e:
            try:
                mask_response = get_mask("2", device_id, device_type, bucket_name, coverage_percents)
            except Exception as e:
                print(e, device_id)
                continue
    print(statistics.mean(coverage_percents))
    counts, edges, bars = plt.hist(coverage_percents, bins=15)
    plt.bar_label(bars)
    plt.show()


def get_mask(ct, device_id, device_type, bucket_name, coverage_percents):
    key = f'{device_id}/calculated_transformations{ct}/{device_id}/mapped_mask_matrix{device_type}_{device_id}.npy'
    mask_response = s3client.get_object(Bucket=bucket_name, Key=key)
    mask_bytes = io.BytesIO(mask_response["Body"].read())
    mask_bytes.seek(0)
    mask_map = np.load(mask_bytes).astype(np.uint8) * 255
    mask_map = mask_map[100:420, 100:340]
    coverage_percents.append(np.count_nonzero(mask_map) / mask_map.size)


if __name__ == "__main__":
    main()
