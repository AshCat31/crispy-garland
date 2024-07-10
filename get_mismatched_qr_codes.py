import json

import numpy as np
import pandas as pd
from s3_setup import setup_s3


def download_json(deviceId):
    global s3client, bucket_name
    json_response = s3client.get_object(Bucket=bucket_name, Key=f'{deviceId}/data.json')
    json_file_content = json_response['Body'].read().decode('utf-8')  # downloading the json
    return json.loads(json_file_content)


s3client, bucket_name = setup_s3()
mismatched_qr_codes = []
id_file = np.genfromtxt("unique_ids.csv", delimiter=",", skip_header=1, dtype=str)
for i, line in enumerate(id_file):
    try:
        id = line[1]
        if id.startswith("E66"):
            dev_type = "head"
        elif id.startswith("100000"):
            dev_type = "hub"
        else:
            continue
        data_content = download_json(id)
        qr_dev_type = {"0103": "hub", "0102": "head"}[data_content['qr_code'].split("-")[1]]
        if dev_type != qr_dev_type:
            mismatched_qr_codes.append([id, data_content["serial_number"], data_content['qr_code'].split(";")[0]])
        if i % 200 == 0:
            print(i)
    except:
        continue
df_out = pd.DataFrame(mismatched_qr_codes)
fn_out = "mismatched_qr_codes.csv"
df_out.to_csv(fn_out)
print("Total mismatches:", len(mismatched_qr_codes))
