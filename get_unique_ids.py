import pandas as pd
import boto3
import numpy as np

import timeit

def get_devices():
    S3_BUCKET = "kcam-calibration-data"
    s3_paginator = boto3.client('s3').get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': S3_BUCKET}
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
    # for filename in np.transpose(everything)[0]:
    #     id = filename.split("/")[0]
    #     if id not in unique_ids:
    #         unique_ids.append(id)
    # unique_devices = [d['Key'].split("/")[0] for d in output if d['Key'][-10:] == '6_inch.png']  # 6in only
    unique_ids = np.unique(np.array([fn.split("/")[0].split(".")[0] for fn in np.transpose(everything)[0]]))
    df_out = pd.DataFrame(unique_ids)
    fn_out = "unique_ids.csv"
    df_out.to_csv(fn_out)


if __name__ == "__main__":
    print(timeit.timeit("main()",globals=globals(),number=1))
