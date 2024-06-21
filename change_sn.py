import json
import os


def download_json(deviceId):
    output = os.system(
        f'aws s3 cp s3://kcam-calibration-data/{deviceId}/data.json ~/S3bucket/{deviceId}/data.json')
    if output != 0:
        return False
    return True

def upload_json(deviceId):
    output = os.system(
        f'aws s3 cp ~/S3bucket/{deviceId}/data.json s3://kcam-calibration-data/{deviceId}/data.json')
    if output != 0:
        return False
    return True

with open("/home/canyon/Test_Equipment/IDs_to_change_SNs.txt", "r") as id_file:
    for line in id_file:
        device = line.split()
        id = device[0]
        download_json(id)
        try:
            with open(f"../S3bucket/{id}/data.json", "r") as json_file:
                data_content = json.load(json_file)
        except FileNotFoundError:
            print(id, "s3 json doesn't exist")
            continue
        print("Old SN is:", data_content['serial_number'])
        new_sn = input("New SN: ")
        if new_sn:
            data_content['serial_number'] = new_sn

            with open(f"../S3bucket/{id}/data.json", "w") as json_file:
                json.dump(data_content, json_file)

            try:
                upload_json(id)
            except FileNotFoundError:
                print(id, "local json doesn't exist")
                continue