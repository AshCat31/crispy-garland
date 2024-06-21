import numpy as np

def get_matches(sn, target_idx, all_devices):
    where = np.where(np.transpose(all_devices)==sn)
    idx = where[1]
    if len(idx>0):
        sns_and_ids.append(" ".join(np.swapaxes(all_devices[idx],0,1)[target_idx]))
    else:
        idx = np.where(np.swapaxes(all_devices,0,1)==" "+sn)[1]
        if len(idx>0):
            sns_and_ids.append(" ".join(np.swapaxes(all_devices[idx],0,1)[target_idx]))
        else:
            if target_idx == 0:  # if finding ID, try prepending a 0
                idx = np.where(np.swapaxes(all_devices,0,1)=="0"+sn)[1]
                if len(idx>0):
                    sns_and_ids.append(" ".join(np.swapaxes(all_devices[idx],0,1)[target_idx]))
                else:
                    sns_and_ids.append("none")
            else:
                sns_and_ids.append("none")

output_path = "/home/canyon/Test_Equipment/SNs_with_IDs.txt"
input_path = "/home/canyon/Test_Equipment/SNs_to_get.txt"
id_list_path = "/home/canyon/Test_Equipment/all_devices.csv"

sns_and_ids = []
# can do ID->ID or SN->SN to get which of a list are in s3
target_idx = {"ID":0,"SN":1}[input("Getting [ID] or [SN]?\n").upper()]

with open(input_path, "r") as in_file:
    sn_list = in_file.read().split("\n")

all_devices = np.genfromtxt(id_list_path, delimiter=",", skip_header=1, dtype=str)
for sn in sn_list:
    get_matches(sn, target_idx, all_devices)
    
with open(output_path, "w") as out_file:
    out_file.write("\n".join(sns_and_ids))
