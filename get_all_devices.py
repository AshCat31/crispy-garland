import csv
from datetime import datetime as dt
import json
import logging
import timeit

import numpy as np

from s3_setup import S3Setup

def checkPath(file_path):
    """Checks whether path exists in s3."""
    result = s3client.list_objects(Bucket=bucket_name, Prefix=file_path)
    return 'Contents' in result   # if anything is found, the file must exist


def get_sn(device_id):
    json_path = f'{device_id}/data.json'
    js_serial_number = "none"
    if checkPath(json_path):  # Checking that the data.json exists
        try:
            json_response = s3client.get_object(Bucket=bucket_name, Key=json_path)
            json_file_content = json_response['Body'].read().decode('utf-8')
            data_content = json.loads(json_file_content)
            js_serial_number = data_content['serial_number']
        except:
            pass
    return js_serial_number


def get_date(id, filenames, csv_data, fileids):
    date = "none"
    try:
        idx = np.where(filenames == id + "/6_inch.png")[0][0]
    except IndexError:  # get another file's last modi instead, which starts with same id
        idx = None
        for i in range(len(fileids)):  # can't get np method to work
            if fileids[i] == id:
                idx = i
                break
    if idx is not None:
        date_info = dt.strptime(csv_data[idx][2][:-6], '%Y-%m-%d %H:%M:%S')
        date = f"{str(date_info.month):0>2}/{str(date_info.day):0>2}/{str(date_info.year):0>4}"
    return date


def main():
    everything = []
    csv_data = np.genfromtxt("kcam-calibration-data-keys.csv", delimiter=",", skip_header=1, dtype=str)
    filenames = csv_data[:,1]
    fileids = [file.split("/")[0].split(".")[0] for file in filenames]
    all_ids = np.genfromtxt("unique_ids.csv", delimiter=',', skip_header=1, usecols=-1, dtype=str)

    for idx, id in enumerate(all_ids):
        try:
            sn = get_sn(id)
            last_modi = get_date(id, filenames, csv_data, fileids)
            everything.append([id, sn, last_modi])
        except Exception as e:
            print("error", id, e)
        if idx % 400 == 0:  # log every 400th to show progress
            print(idx)
    with open("all_devices.csv", "w", newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(everything)


if __name__ == "__main__":
    global logger, s3client, bucket_name
    logger = logging.getLogger(__name__)
    log_format = '%(levelname)-6s: %(message)s'
    logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.basicConfig(level=logging.WARN, format=log_format)

    s3s = S3Setup()
    s3client, bucket_name = s3s()

    # main()
    print(timeit.timeit("main()", number=1, globals=globals()))
