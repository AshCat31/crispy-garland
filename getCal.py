import os

_device_file = "/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt"


def download_device(deviceId):
    print("Downloading", deviceId)
    output = os.system(f"aws s3 cp s3://kcam-calibration-data/{deviceId} ~/S3bucket/{deviceId} --recursive --only-show-errors")
    return output == 0


def main():
    with open(_device_file) as csvFile:
        content = csvFile.read()
        deviceList = [line.split() for line in content.split("\n")]
        for device in deviceList:
            download_device(device[0])


if __name__ == "__main__":
    main()
