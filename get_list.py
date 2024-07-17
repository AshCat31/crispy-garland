import boto3
import pandas as pd


def get_date_range(input_str):
    while True:
        try:
            return tuple([int(d) for d in input(input_str).split("-")])
        except ValueError:
            print("Invalid input.")


def filter_by_date(device_list, months=(1, 12), days=(1, 31)):
    month_range = range(months[0], months[-1] + 1)
    day_range = range(days[0], days[-1] + 1)
    devices_per_day = {}
    filtered_devices = []
    for m in month_range:
        for d in day_range:
            if (d == 31 and m in (4, 6, 9, 11)) or (d == 30 and m == 2):
                break
            key = f"{str(m):0>2}/{str(d):0>2}"
            devices_per_day[key] = 0
    for d in device_list:
        date = d["LastModified"].date()
        if date.month in month_range and date.day in day_range:
            filtered_devices.append(d)
            key = f"{str(date.month):0>2}/{str(date.day):0>2}"
            if key in devices_per_day.keys():
                devices_per_day[key] += 1
            else:
                devices_per_day[key] = 1
    return filtered_devices, devices_per_day


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
    df_out = pd.DataFrame(output)
    fn_out = "kcam-calibration-data-keys.csv"
    df_out.to_csv(fn_out)
    return output

def filter_devices(devices):
    unique_devices = [d for d in devices if d['Key'][-10:] == '6_inch.png' and d["LastModified"].date().year == 2024]
    filtered_devices = []
    devices_per_day = {}
    month_range = ()

    while month_range != (0,):
        month_range = get_date_range("Enter range of months, ex 1-2 for Jan-Feb, 3 for just March, or 0 to quit\n")
        if month_range == (0,):
            break
        elif len(month_range) == 2:
            day_range = (1, 31)
        elif len(month_range) == 1:
            day_range = get_date_range("Enter range of dates, ex 10-15 or 4\n")
        else:
            print("Invalid input")
            continue

        filtered_devices, devices_per_day = filter_by_date(unique_devices, month_range, day_range)
        for month_day, count in dict(sorted(devices_per_day.items())).items():
            print(f"{month_day}:\t{count}")
        devices_per_day = {}
        print("Total devices:", len(filtered_devices))
        filtered_devices = []

def main():
    devices = get_devices()
    filter_devices(devices)

if __name__ == "__main__":
    main()
