import os

_device_file = '/home/canyon/Test_Equipment/QA_ids.txt'


def upload_device(deviceId):
    print("Uploading", deviceId)
    output = os.system(
                f'aws s3 cp ~/S3bucket/{deviceId} s3://kcam-calibration-data/{deviceId} --recursive --only-show-errors')
    if output != 0:
        return False
    return True

def main():
    with open(_device_file) as csvFile:
        content = csvFile.read()  # allows tabs and spaces
        deviceList = [line.split() for line in content.split('\n')]
        for device in deviceList:
            upload_device(device[0])


if __name__ == "__main__":
    main()
