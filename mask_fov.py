import boto3
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    global x_diffs, y_diffs
    device_list = []
    x_diffs = []
    y_diffs = []
    doc_path = '/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt'
    with open(doc_path, 'r') as file:
        for line in file:
            device_list.append(line.split()[0])
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
    for id in device_list:
        try:
            find_extreme_indices(get_mask("", id, "_hydra", "kcam-mosaic-calibration"))
        except:
            find_extreme_indices(get_mask("2", id, "_hydra", "kcam-mosaic-calibration"))
    fig, axs = plt.subplots(1, 2)
    bins = 9

    # Plot histograms and get counts
    counts_x, bins_x, _ = axs[0].hist(x_diffs, bins=bins, histtype='bar')
    counts_y, bins_y, _ = axs[1].hist(y_diffs, bins=bins, histtype='bar')

    # Set number of ticks on x-axis to match bins
    axs[0].set_xticks([int(i) for i in bins_x])
    axs[1].set_xticks([int(i) for i in bins_y])

    # Annotate bars in the first subplot (x_diffs)
    for count, bin_edge in zip(counts_x, bins_x):
        if count != 0:  # Ignore zero counts
            axs[0].annotate(int(count), xy=(bin_edge + 1, count), xytext=(0, 3),
                            textcoords='offset points', ha='center', va='bottom')

    # Annotate bars in the second subplot (y_diffs)
    for count, bin_edge in zip(counts_y, bins_y):
        if count != 0:  # Ignore zero counts
            axs[1].annotate(int(count), xy=(bin_edge + 1, count), xytext=(0, 3),
                            textcoords='offset points', ha='center', va='bottom')

    # Set labels and titles for each subplot
    axs[0].set_xlabel('X Differences')
    # axs[0].set_ylabel('Frequency')
    # axs[0].set_title('Histogram of X Differences')

    axs[1].set_xlabel('Y Differences')
    # axs[1].set_ylabel('Frequency')
    # axs[1].set_title('Histogram of Y Differences')

    # Ensure the subplots are properly spaced
    plt.tight_layout()

    # Show the plot
    plt.show()


# maximize/minimize mask maps
def get_mask(ct, device_id, device_type, bucket_name):
    global mask_map
    # inner_dir = get_inner_dir(bucket_name, f'{device_id}/calculated_transformations{ct}/')
    key = f'{device_id}/calculated_transformations{ct}/{device_id}/mapped_mask_matrix{device_type}_{device_id}.npy'
    # print(key)
    local_directory = '/home/canyon/S3bucket/'
    try:
        mask_map = np.load(os.path.join(local_directory, key)).astype(np.uint8) * 255
    except:  # currently not working?? now?
        try:
            os.makedirs(os.path.join(local_directory, f'{device_id}/calculated_transformations{ct}/{device_id}'))
        except FileExistsError:
            # try:
            #     os.mkdir(os.path.join(local_directory, f'{device_id}/calculated_transformations{ct}'))
            #     os.mkdir(os.path.join(local_directory, f'{device_id}/calculated_transformations{ct}/{device_id}'))
            # except FileExistsError:
            #     try:
            #         os.mkdir(os.path.join(local_directory, f'{device_id}/calculated_transformations{ct}/{device_id}'))
            #     except:
            pass
        s3client.download_file(Bucket=bucket_name, Key=key,
                               Filename=os.path.join(local_directory, key))
    return np.load(os.path.join(local_directory, key)).astype(np.uint8)


def find_extreme_indices(array):
    nonzero_indices = np.nonzero(array)

    if len(nonzero_indices[0]) == 0:
        print("Array has no non-zero elements.")
        return

    # Find extreme indices
    highest_index = np.argmax(nonzero_indices[0])
    lowest_index = np.argmin(nonzero_indices[0])
    leftmost_index = np.argmin(nonzero_indices[1])
    rightmost_index = np.argmax(nonzero_indices[1])

    # Print differences
    highest_lowest_diff = np.abs(nonzero_indices[0][highest_index] - nonzero_indices[0][lowest_index])
    leftmost_rightmost_diff = np.abs(nonzero_indices[1][rightmost_index] - nonzero_indices[1][leftmost_index])

    # print(f"Highest index: ({nonzero_indices[0][highest_index]}, {nonzero_indices[1][highest_index]})")
    # print(f"Lowest index: ({nonzero_indices[0][lowest_index]}, {nonzero_indices[1][lowest_index]})")
    # print(f"Leftmost index: ({nonzero_indices[0][leftmost_index]}, {nonzero_indices[1][leftmost_index]})")
    # print(f"Rightmost index: ({nonzero_indices[0][rightmost_index]}, {nonzero_indices[1][rightmost_index]})")
    y_diffs.append(highest_lowest_diff)
    x_diffs.append(leftmost_rightmost_diff)


if __name__ == '__main__':
    main()
