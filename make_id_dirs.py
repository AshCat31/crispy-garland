import os

device_file = '/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt'
path = '/home/canyon/S3bucket'

with open(device_file, 'r') as file:
    for line in file:
        # Strip any whitespace (like newlines) from the line
        folder_name = line.strip()
        
        # Construct full path for the new folder
        folderpath = os.path.join(path, folder_name)
        
        # Check if the folder already exists
        if os.path.exists(folderpath):
            print(f"Folder already exists: {folderpath}")
        else:
            # Create the folder
            os.makedirs(folderpath)
            print(f"Created folder: {folderpath}")