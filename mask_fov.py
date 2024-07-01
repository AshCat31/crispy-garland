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
    # doc_path = '/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt'
    good_path = "/home/canyon/Test_Equipment/crispy-garland/integrated_heads_not_p3.txt"
    bad_path = "/home/canyon/Test_Equipment/crispy-garland/heads_roi_rma.txt"
    with open(good_path, 'r') as file:
        for line in file:
            good_list.append(line.split()[0])
    with open(bad_path, 'r') as file:
        for line in file:
            bad_list.append(line.split()[0])
    cred = boto3.Session().get_credentials()
    ACCESS_KEY = cred.access_key
    SECRET_KEY = cred.secret_key
    SESSION_TOKEN = cred.token
    global s3client
    s3client = boto3.client('s3',
                            aws_access_key_id=ACCESS_KEY,
                            aws_secret_access_key=SECRET_KEY,
                            aws_session_token=SESSION_TOKEN,
                            )
    img = np.zeros((460, 385)).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    fig, axs = plt.subplots(1,1)
    h_sorted_masks = []
    for id in bad_list:
        plot_mask(id, img, (255,50,255), h_sorted_masks)
    #     # print(idx)
    print(h_sorted_masks)
    for line in sorted(h_sorted_masks, key=lambda x: -x[0]):
        # list.append([height, mask_edges_contours, color2])
        cv2.drawContours(img, line[1], -1, line[2], 1)
    # for id in good_list:
    #     plot_mask(id, img, (0,255,0))
    # fig, axs = plt.subplots(1,3)
    axs.imshow(img[65:,40:,], cmap='grey')
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
    axs.axis('off')
    print(min(y_diffs), max(y_diffs), min(x_diffs), max(x_diffs))
    plt.tight_layout()
    plt.show()

def normalize(x, min_value=321, max_value=346):
    return (x - min_value) / (max_value - min_value)

def get_mask(ct, device_id, device_type, bucket_name):
    global mask_map
    key = f'{device_id}/calculated_transformations{ct}/{device_id}/mapped_mask_matrix{device_type}_{device_id}.npy'
    local_directory = '/home/canyon/S3bucket/'
    try:
        mask_map = np.load(os.path.join(local_directory, key)).astype(np.uint8) * 255
    except:  # currently not working?? now?
        try:
            os.makedirs(os.path.join(local_directory, f'{device_id}/calculated_transformations{ct}/{device_id}'))
        except FileExistsError:
            pass
        s3client.download_file(Bucket=bucket_name, Key=key,
                               Filename=os.path.join(local_directory, key))
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
    mask_edges = cv2.Canny(mask_map*255, 30, 200)
    mask_edges_contours, _ = cv2.findContours(mask_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    color2 = (color[0]-255*normalize(height), color[1], color[2]*normalize(height))
    if list is not None:
        list.append([height, mask_edges_contours, color2])
    else:
        cv2.drawContours(img, mask_edges_contours, -1, color2, 1)
    # x = mask_edges_contours[0]
    # if max(x[:,0,1]) == 439:
        # print((x[:,0,1]), id)

if __name__ == '__main__':
    main()
