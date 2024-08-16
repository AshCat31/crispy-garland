import os
import numpy as np
import matplotlib.pyplot as plt
import s3_utils as s3u
from s3_setup import S3Setup


def vector_check(id):
    # Assuming s3u is a module for interacting with S3 and is already imported
    device_type, device_idx = s3u.get_device_type_and_idx(id)
    key = f"{id}/rgb_{device_idx}_9element_coord.npy"
    local_directory = os.path.join(dir_path, "S3bucket/")
    os.makedirs(os.path.join(local_directory, id), exist_ok=True)
    x = s3u.load_numpy_array_from_s3(key)

    if x is not None and x.size > 0:
        coords = np.array(x)  # Ensure x is a numpy array
        return coords
    else:
        return np.array([])  # Return an empty array if no coordinates


def calculate_vector_properties(coords):
    # Compute vectors between consecutive points
    vectors = np.diff(coords, axis=0)

    # Compute magnitudes
    magnitudes = np.linalg.norm(vectors, axis=1)

    # Compute angles (in degrees)
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))

    return vectors, magnitudes, angles


def plot_histograms(all_vectors, title_prefix):
    num_vectors = len(all_vectors)

    # Set up a 4x4 grid for subplots
    fig, axs = plt.subplots(4, 4, figsize=(15, 12))

    # Flatten the 2D array of axes to make indexing easier
    axs = axs.flatten()
    with open("autopass.npy", "wb") as f:
        np.save(f, np.asarray(all_vectors))

    for i in range(8):
        if all_vectors[i]:
            magnitudes, angles = zip(*all_vectors[i])

            # Plot histogram for magnitudes
            axs[2 * i].hist(magnitudes, bins=10, color="blue", edgecolor="black", alpha=0.7)
            axs[2 * i].set_title(f"{title_prefix} Vector {i+1} Magnitudes")

            # Plot histogram for angles
            axs[2 * i + 1].hist(angles, bins=10, color="orange", edgecolor="black", alpha=0.7)
            axs[2 * i + 1].set_title(f"{title_prefix} Vector {i+1} Angles")
        else:
            # Hide unused subplots
            axs[2 * i].set_visible(False)
            axs[2 * i + 1].set_visible(False)

    # Hide any remaining unused subplots
    for j in range(2 * num_vectors, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def main():
    s3c = S3Setup()
    s3client, bucket_name = s3c()
    global dir_path
    dir_path = os.path.dirname(os.path.abspath(__file__))
    device_file = os.path.join(dir_path, "autopass.txt")

    # Read device list
    with open(device_file) as csvFile:
        content = csvFile.read()
        deviceList = [line.split() for line in content.split("\n")]

    all_vectors = [[] for _ in range(8)]  # List to store vectors for each of the 8 positions

    # Collect coordinates from all devices
    for device in deviceList:
        coords = vector_check(device[0])
        if coords.size > 0:
            _, magnitudes, angles = calculate_vector_properties(coords)

            # Store vectors for each of the 8 positions
            for i in range(len(magnitudes)):
                all_vectors[i].append((magnitudes[i], angles[i]))

    # Plot histograms for each vector
    plot_histograms(all_vectors, "Device")


if __name__ == "__main__":
    main()
