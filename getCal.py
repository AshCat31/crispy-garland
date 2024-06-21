import os
import csv

# _device_file = "/home/canyon/Test_Equipment/provisioning/docs.csv"
_device_file = '/home/canyon/Test_Equipment/QA_ids.txt'


def putFile(deviceId, destination):
    output = os.system(f'mv ~/images/{deviceId}* ~/S3bucket/{destination}/')


def downloadFiles(deviceId):
    output = os.system(
        f'aws s3 cp s3://kcam-calibration-data/{deviceId} ~/S3bucket/{deviceId} --recursive --only-show-errors')
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
            f'aws s3 cp s3://kcam-calibration-data/{serialNumber} ~/S3bucket/{serialNumber} --recursive --only-show-errors')
        putFile(cameraId, serialNumber)
    else:
        output3 = os.system(
            f'aws s3 cp s3://kcam-calibration-data/{cameraId} ~/S3bucket/{cameraId} --recursive --only-show-errors')
        putFile(cameraId, cameraId)
    return True

def download_device(device):
    print("Downloading", "\t ".join(device))
    if len(device) == 3:
        downloadHub(device[2], device[0])
    else:
        downloadFiles(device[0])

def main():
    with open(_device_file) as csvFile:
        # InDeviceId = str(input("Input the device Id. Input nothing if you wish to use docs.csv. "))
        InDeviceId = ''
        DeviceId = InDeviceId
        # deviceList = csv.reader(csvFile, delimiter='\t')
        
        content = csvFile.read()  # allows tabs and spaces
        deviceList = []
        for line in content.split("\n"):
            deviceList.append(line.split())

        for device in deviceList:
            if len(DeviceId) > 0:
                downloadFiles(DeviceId)
                break
            elif DeviceId == "\0":
                continue
            download_device(device)


if __name__ == "__main__":
    main()
