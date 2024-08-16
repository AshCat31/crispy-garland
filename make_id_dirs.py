import os

device_file = "/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt"
path = "/home/canyon/S3bucket"

with open(device_file, "r") as file:
    for line in file:
        folder_name = line.split()[0].strip()
        folderpath = os.path.join(path, folder_name, "calculated_transforms", folder_name)
        if os.path.exists(folderpath):
            print(f"Folder already exists: {folderpath}")
        else:
            os.makedirs(folderpath)
            print(f"Created folder: {folderpath}")
