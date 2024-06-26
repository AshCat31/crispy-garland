import csv
import datetime as dt
import json
import logging

import boto3
import numpy as np


# Function to check if a file exists in S3
def checkPath(file_path):
    result = s3client.list_objects(Bucket=_bucket_name,
                                   Prefix=file_path)  # This pulls all files in S3 with the 'File Path' Prefix
    exists = False
    # Because the paths Defined above are file paths and not folder paths if anything is found at that path it can
    # only be the file, therefore it exists
    if 'Contents' in result:
        exists = True
    return exists


def get_sn(device_id):
    folder_path = f'{device_id}/'
    json_path = f'{device_id}/data.json'
    js_serial_number = "none"
    # Checks:
    if checkPath(folder_path):  # Checking that the folder exists
        if checkPath(json_path):  # Checking that the data.json exists
            try:
                json_response = s3client.get_object(Bucket=_bucket_name, Key=f'{device_id}/data.json')
                json_file_content = json_response['Body'].read().decode('utf-8')  # downloading the json
                data_content = json.loads(json_file_content)
                js_serial_number = data_content['serial_number']
            except:
                pass
    return js_serial_number


def get_date(id, filenames, file_list, fileids):
    date = "none"
    try:
        idx = np.where(filenames == id + "/6_inch.png")[0][0]
        date_info = dt.datetime.strptime(file_list[idx][2][:-6], '%Y-%m-%d %H:%M:%S')
        date = f"{str(date_info.month):0>2}/{str(date_info.day):0>2}/{str(date_info.year):0>4}"
        return date
    except IndexError:  # get another file's last modi instead, where starts with same id
        idx = None
        for i in range(len(fileids)):  # can't get np to work
            if fileids[i] == id:
                idx = i
                break
        if idx:
            date_info = dt.datetime.strptime(file_list[idx][2][:-6], '%Y-%m-%d %H:%M:%S')
            date = f"{str(date_info.month):0>2}/{str(date_info.day):0>2}/{str(date_info.year):0>4}"
        return date


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
    # defs
    global _bucket_name
    _bucket_name = 'kcam-calibration-data'
    everything = []
    csv_data = np.genfromtxt("kcam-calibration-data-keys.csv", delimiter=",", skip_header=1, dtype=str)
    file_list = [line for line in csv_data]
    filenames = np.swapaxes(file_list, 0, 1)[1]
    fileids = [file.split("/")[0].split(".")[0] for file in filenames]
    with open("unique_ids.csv", 'r') as file_ids:
        csv_ids = csv.reader(file_ids, delimiter=',')
        next(csv_ids, None)
        all_ids = [line[-1] for line in csv_ids]
        for idx, id in enumerate(all_ids):
            try:
                sn = get_sn(id)
                last_modi = get_date(id, filenames, file_list, fileids)
                everything.append([id, sn, last_modi])
            except Exception as e:
                print("error", id, e)
            if idx % 400 == 0:  # log every 400th to show progress
                print(idx)
    with open("all_devices.csv", "w", newline='') as out_file:
        # with open("a_device.csv", "w", newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(everything)


if __name__ == "__main__":
    main()
    # print(timeit.timeit("main()", number=1, globals=globals()))
    # print(timeit.timeit('get_date("E660D05113338132")', globals=globals(), number=10))
    # print(timeit.timeit('get_date("E661385283302332")', globals=globals(), number=10))
