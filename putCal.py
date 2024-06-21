import csv
import os

# _device_file = "/home/canyon/Test_Equipment/provisioning/docs.csv"
_device_file = '/home/canyon/Test_Equipment/QA_ids.txt'


def downloadFiles(deviceId):
    output = os.system(
        f'aws s3 cp ~/S3bucket/{deviceId} s3://kcam-calibration-data/{deviceId} --recursive --only-show-errors')
    if output != 0:
        return False
    return True


def downloadHub(cameraId, serialNumber):
    output = os.system(f'aws s3 ls s3://kcam-calibration-data/{cameraId}')

    if output != 0:
        print("cameraId Failed trying serial number")
        output2 = os.system(f'aws s3 ls s3://kcam-calibration-data/{serialNumber}')
        if output2 != 0:
            return False
        output3 = os.system(
            f'aws s3 cp ~/S3bucket/{serialNumber} s3://kcam-calibration-data/{serialNumber} --recursive --only-show-errors')
    else:
        output3 = os.system(
            f'aws s3 cp ~/S3bucket/{cameraId} s3://kcam-calibration-data/{cameraId}  --recursive --only-show-errors')
    return True


def main():
    with open(_device_file) as csvFile:
        # InDeviceId = str(input("Input the device Id. Input nothing if you wish to use docs.csv. "))
        InDeviceId = ''  # always use the file
        DeviceId = InDeviceId
        # deviceList = csv.reader(csvFile, delimiter='\t')
        content = csvFile.read()
        deviceList = []
        for line in content.split("\n"):
            deviceList.append(line.split())
        for device in deviceList:
            if len(DeviceId) > 0:
                downloadFiles(DeviceId)
                break
            elif DeviceId == "\0":
                continue
            upload_device(device)

def upload_device(device):
    print("Uploading", "\t ".join(device))
    if len(device) == 3:
        downloadHub(device[2], device[0])
    else:
        downloadFiles(device[0])


if __name__ == "__main__":
    main()
