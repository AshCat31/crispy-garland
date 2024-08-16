"""Generates csv of unique device IDs"""

import boto3
import numpy as np
import pandas as pd


def get_devices():
    S3_BUCKET = "kcam-calibration-data"
    s3_paginator = boto3.client("s3").get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": S3_BUCKET}
    page_iterator = s3_paginator.paginate(**operation_parameters)
    print("Finding S3 keys...")
    output = list()
    for i, page in enumerate(page_iterator):
        for j, record in enumerate(page["Contents"]):
            output.append(record)
    return output


def main():
    output = get_devices()
    everything = np.asarray([list(line.values()) for line in output])
    unique_ids = []
    unique_ids = np.unique(np.array([fn.split("/")[0].split(".")[0] for fn in np.transpose(everything)[0]]))
    df_out = pd.DataFrame(unique_ids)
    fn_out = "unique_ids.csv"
    df_out.to_csv(fn_out)


if __name__ == "__main__":
    main()
