import boto3
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image as PImage


def main():
    global x_diffs, y_diffs
    device_list = []
    good_list = []
    bad_list = []
    x_diffs = []
    y_diffs = []
    doc_path = "/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt"
    good_path = "/home/canyon/Test_Equipment/crispy-garland/integrated_heads_not_p3.txt"
    bad_path = "/home/canyon/Test_Equipment/crispy-garland/heads_roi_rma.txt"
    with open(good_path, "r") as file:
        for line in file:
            good_list.append(line.split()[0])
    with open(doc_path, "r") as file:
        for line in file:
            device_list.append(line.split()[0])
    with open(bad_path, "r") as file:
        for line in file:
            bad_list.append(line.split()[0])
    cred = boto3.Session().get_credentials()
    ACCESS_KEY = cred.access_key
    SECRET_KEY = cred.secret_key
    SESSION_TOKEN = cred.token
    global s3client
    s3client = boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        aws_session_token=SESSION_TOKEN,
    )
    device_rois = [
        "/home/canyon/Test_Equipment/head_alignment_test/auto_port0_one.npy",
        "/home/canyon/Test_Equipment/head_alignment_test/auto_port1_one.npy",
        "/home/canyon/Test_Equipment/head_alignment_test/auto_port2_one.npy",
    ]
    width = 390
    height = 460
    fig, axs = plt.subplots(1, 1)
    # fig, axs = plt.subplots(1,3)
    colors = ["cyan", "brown", "magenta", "blue", "green", "yellow", "orange", "red"]
    colors2 = ["gray", "lightgrey", "darkgrey"]
    cidx2 = 0
    for roi_file in device_rois:
        cidx = 0
        rois = np.load(roi_file)
        for indx, roi in enumerate(rois):
            if (
                ("port1" in roi_file and (colors[cidx % 8] == "cyan" or colors[cidx % 8] == "yellow"))
                or ("port2" in roi_file and (colors[cidx % 8] == "cyan" or colors[cidx % 8] == "red"))
                or (
                    "port0" in roi_file
                    and (colors[cidx % 8] == "cyan" or colors[cidx % 8] == "red" or colors[cidx % 8] == "blue")
                )
            ):
                roi_x, roi_y = width - roi[:, :, 0], height - roi[:, :, 1]
                axs.plot(roi_x, roi_y, "o", color=colors2[cidx2 % 8], markersize=5)
            cidx += 1
        cidx2 += 1
    img_padded = np.zeros((height, width)).astype(np.uint8)
    # img_padded[115:435, 40:280] = np.asarray(img.rotate(180))
    img = img_padded
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h_sorted_masks = []
    for id in bad_list:
        plot_mask(id, img, (255, 50, 255), h_sorted_masks)
    # h_sorted_masks = sorted(h_sorted_masks, key=lambda x: -x[0])
    # for line in h_sorted_masks:
    # # for line in h_sorted_masks[:int(.5*len(h_sorted_masks))]:
    #     cv2.drawContours(img, line[1], -1, line[2], 1)
    for id in device_list:
        plot_mask(id, img, (255, 255, 240))
    for id in good_list:
        plot_mask(id, img, (0, 255, 0))
    axs.imshow(img[65:, 40:], cmap="grey")
    # axs[2].imshow(img, cmap='grey')
    # bins = 9
    # # Plot histograms and get counts
    # counts_x, bins_x, _ = axs[0].hist(x_diffs, bins=bins, histtype='bar')
    # counts_y, bins_y, _ = axs[1].hist(y_diffs, bins=bins, histtype='bar')

    # # Set number of ticks on x-axis to match bins
    # axs[0].set_xticks([int(i) for i in bins_x])
    # axs[1].set_xticks([int(i) for i in bins_y])

    # # Annotate bars in the first subplot (x_diffs)
    # for count, bin_edge in zip(counts_x, bins_x):
    #     if count != 0:  # Ignore zero counts
    #         axs[0].annotate(int(count), xy=(bin_edge + 1, count), xytext=(0, 3),
    #                         textcoords='offset points', ha='center', va='bottom')

    # # Annotate bars in the second subplot (y_diffs)
    # for count, bin_edge in zip(counts_y, bins_y):
    #     if count != 0:  # Ignore zero counts
    #         axs[1].annotate(int(count), xy=(bin_edge + 1, count), xytext=(0, 3),
    #                         textcoords='offset points', ha='center', va='bottom')

    # axs[0].set_xlabel('X Differences')
    # axs[1].set_xlabel('Y Differences')
    # axs[2].axis('off')
    axs.axis("off")
    plt.tight_layout()
    plt.show()


def normalize(x, min_value=321, max_value=346):
    return (x - min_value) / (max_value - min_value)


def get_mask(ct, device_id, device_type, bucket_name):
    global mask_map
    key = f"{device_id}/calculated_transformations{ct}/{device_id}/mapped_mask_matrix{device_type}_{device_id}.npy"
    local_directory = "/home/canyon/S3bucket/"
    try:
        mask_map = np.load(os.path.join(local_directory, key)).astype(np.uint8) * 255
    except:  # currently not working?? now?
        try:
            os.makedirs(
                os.path.join(
                    local_directory,
                    f"{device_id}/calculated_transformations{ct}/{device_id}",
                )
            )
        except FileExistsError:
            pass
        s3client.download_file(Bucket=bucket_name, Key=key, Filename=os.path.join(local_directory, key))
    return np.load(os.path.join(local_directory, key)).astype(np.uint8)


def find_extreme_indices(array):
    nonzero_indices = np.nonzero(array)
    if len(nonzero_indices[0]) == 0:
        print("Array has no non-zero elements.")
        return

    highest_index = np.argmax(nonzero_indices[0])
    lowest_index = np.argmin(nonzero_indices[0])
    leftmost_index = np.argmin(nonzero_indices[1])
    rightmost_index = np.argmax(nonzero_indices[1])

    highest_lowest_diff = np.abs(nonzero_indices[0][highest_index] - nonzero_indices[0][lowest_index])
    leftmost_rightmost_diff = np.abs(nonzero_indices[1][rightmost_index] - nonzero_indices[1][leftmost_index])

    y_diffs.append(highest_lowest_diff)
    x_diffs.append(leftmost_rightmost_diff)
    return highest_lowest_diff


def plot_mask(id, img, color, list=None):
    try:
        mask_map = get_mask("", id, "_hydra", "kcam-mosaic-calibration")
    except:
        mask_map = get_mask("2", id, "_hydra", "kcam-mosaic-calibration")
    height = find_extreme_indices(mask_map)
    mask_edges = cv2.Canny(mask_map * 255, 30, 200)
    mask_edges_contours, _ = cv2.findContours(mask_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    color2 = color
    # color2 = (color[0]-255*normalize(height), color[1], color[2]*normalize(height))
    if list is not None:
        list.append([height, mask_edges_contours, color2])
    else:
        cv2.drawContours(img, mask_edges_contours, -1, color2, 1)
    # x = mask_edges_contours[0]
    # if max(x[:,0,1]) == 439:
    # print((x[:,0,1]), id)


if __name__ == "__main__":
    main()
