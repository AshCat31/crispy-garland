import numpy as np


def get_matches(sn, id_idx, all_devices, matched_ids):
    idx = np.where(all_devices.transpose() == sn)[1]
    if len(idx > 0):
        matched_ids.append(
            np.concatenate((np.swapaxes(all_devices[idx], 0, 1)[id_idx], np.swapaxes(all_devices[idx], 0, 1)[2])))
    else:
        idx = np.where(all_devices.transpose() == " " + sn)[1]
        if len(idx > 0):
            matched_ids.append(
                np.concatenate((np.swapaxes(all_devices[idx], 0, 1)[id_idx], np.swapaxes(all_devices[idx], 0, 1)[2])))
        else:
            matched_ids.append("none")
    matched_ids = matched_ids[0][0].split()


output_path = "/home/canyon/Test_Equipment/crispy-garland/duplicate_SNs.txt"
id_list_path = "/home/canyon/Test_Equipment/crispy-garland/all_devices.csv"

id_idx = 0
sn_idx = 1
date_idx = 2
matches_dict = {}
all_devices = np.genfromtxt(id_list_path, delimiter=",", skip_header=1, dtype=str)

duplicate_sns = [sn for sn in all_devices.transpose()[sn_idx] if len(np.where(all_devices.transpose() == sn)[0]) > 1]

for sn in duplicate_sns:
    matched_ids = []
    get_matches(sn, id_idx, all_devices, matched_ids)
    matched_ids = matched_ids[0]
    hub_ct = head_ct = other_ct = 0
    for id in matched_ids:  # don't add if they're different device types
        if id.startswith("100"):
            hub_ct += 1
        elif id.startswith("E66"):
            head_ct += 1
        # else:  # includes dates if on
        #     other_ct +=1 
    if head_ct > 1 or hub_ct > 1 or other_ct > 1:  # if more than 1 of a type
        matches_dict[sn] = matched_ids
with open(output_path, "w") as out_file:
    tab = "\t"
    out_file.write(
        "\n".join(sorted([f"{sn}{tab}{tab.join([str(id) for id in ids])}" for sn, ids in matches_dict.items()])))
