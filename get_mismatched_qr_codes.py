import json
import numpy as np
import pandas as pd

from s3_setup import S3Setup


def load_json(deviceId):
    global s3client, bucket_name
    try:
        json_response = s3client.get_object(
            Bucket=bucket_name, Key=f"{deviceId}/data.json"
        )
    except:
        return {}
    json_file_content = json_response["Body"].read().decode("utf-8")
    return json.loads(json_file_content)


def process_line(line):
    try:
        id = line[1]
        result = process_data(id)
        return result
    except Exception as e:
        print(f"Error processing line {line}: {e}")
    return None


def process_data(id):
    dev_type = get_device_type(id)

    data_content = load_json(id)
    if dev_type is None or "qr_code" not in data_content.keys():
        return None

    qr_code = data_content["qr_code"].split(";")[0]
    if qr_code == "":
        return None

    qr_dev_type = parse_qr_code(qr_code)

    if qr_dev_type is None or dev_type != qr_dev_type:
        return [id, data_content.get("serial_number", ""), qr_code]

    return None


def get_device_type(id):
    if id.startswith("E66"):
        return "head"
    elif id.startswith("100000"):
        return "hub"
    else:
        return None


def parse_qr_code(qr_code):
    try:
        qr_dev_type = {"0103": "hub", "0102": "head"}[qr_code.split("-")[1]]
        return qr_dev_type
    except (KeyError, IndexError):
        return None


def check_qr_code_mismatches(id_file):
    mismatched_qr_codes = []
    for i, line in enumerate(id_file):
        result = process_line(line)
        if result:
            mismatched_qr_codes.append(result)
        if i % 200 == 0:  # gives sense of progress
            print(i)
    return mismatched_qr_codes


def write_to_csv(mismatched_qr_codes, filename):
    df_out = pd.DataFrame(
        mismatched_qr_codes, columns=["ID", "Serial Number", "QR Code"]
    )
    df_out.to_csv(filename, index=False)
    print(f"Total mismatches: {len(mismatched_qr_codes)}")


if __name__ == "__main__":
    s3c = S3Setup()
    s3client, bucket_name = s3c()
    id_file = np.genfromtxt("unique_ids.csv", delimiter=",", skip_header=1, dtype=str)
    mismatched_qr_codes = check_qr_code_mismatches(id_file)
    write_to_csv(mismatched_qr_codes, "mismatched_qr_codes.csv")
