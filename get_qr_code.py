import json
import os


def download_json(deviceId):
    output = os.system(f'aws s3 cp s3://kcam-calibration-data/{deviceId}/data.json ~/S3bucket/{deviceId}/data.json --only-show-errors')
    if output != 0:
        return False
    return True


head_ct = hub_ct = 0
with open("/home/canyon/Test_Equipment/crispy-garland/SNs_with_IDs.txt", "r") as id_file:
    for line in id_file:
        device = line.split()
        id = device[0]
        download_json(id)
        try:
            with open(f"../../S3bucket/{id}/data.json", "r") as json_file:
                data_content = json.load(json_file)
        except FileNotFoundError:
            print(id, "s3 json doesn't exist")
            continue
        dev_type = data_content['qr_code'].split(";")[0]
        if dev_type == "400-0103-02":
            hub_ct += 1
        else:
            head_ct += 1
print("Hubs:", hub_ct)
print("Heads:", head_ct)
