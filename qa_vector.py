import os
import statistics

import boto3
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.cm import ScalarMappable


def vector_plot(magnitude, angle_degrees, color='black', zorder=10):
    angle_radians = np.deg2rad(-angle_degrees + 270)
    x_component = magnitude * np.cos(angle_radians)
    y_component = magnitude * np.sin(angle_radians)
    # ax.arrow(0, 0, x_component, y_component, head_width=1, color=color)
    if color == 'yellow':
        pass
        # ax.plot(x_component, y_component, 'o', color=color, markersize=12, zorder=zorder)
    if color == 'black':
        xs.append(x_component)
        ys.append(y_component)
    elif color != 'red':
        new_xs.append(x_component)
        new_ys.append(y_component)
    # xs.append(x_component)
    # ys.append(y_component)
    # def vector_plot(magnitude, angle_degrees, color='grey'):
    # angle_radians = np.deg2rad(-angle_degrees+270)
    # x_component = magnitude * np.cos(angle_radians)
    # y_component = magnitude * np.sin(angle_radians)
    # ax.arrow(0, 0, x_component, y_component, head_width=1, color=color, label=label)
    # if color == 'grey':
    #     color = 'black'
    #     markersize=5
    # else:
    #     markersize = 10
    # ax.plot(x_component, y_component, 'o', color=color, markersize=markersize)

    # ax[2].annotate(int(angle_degrees+180), xy=(x_component, y_component), xytext=(x_component+0.1, y_component+0.1))


def get_vector(device_id):
    try:
        mask = np.load(os.path.join(local_directory, device_id, f'mapped_mask_matrix_hydra_{device_id}.npy'))
    except FileNotFoundError:
        try:
            s3client.download_file(Bucket=bucket_name,
                                   Key=f'{device_id}/calculated_transformations/{device_id}/mapped_mask_matrix_hydra_{device_id}.npy',
                                   Filename=os.path.join(local_directory, device_id,
                                                         f'mapped_mask_matrix_hydra_{device_id}.npy'))
        except FileNotFoundError:
            print("Error:", device_id, "has no mapped mask matrix")
            return None
    try:
        mask = np.load(os.path.join(local_directory, device_id, f'mapped_mask_matrix_hydra_{device_id}.npy'))
    except FileNotFoundError:
        print("Error:", device_id, "not found")
        return None
    mask = mask.astype(np.uint8) * 255
    therm_x = statistics.mean(np.nonzero(mask)[0])
    therm_y = statistics.mean(np.nonzero(mask)[1])
    therm_cen = (therm_x, therm_y)

    # get vector
    dX = rgb_cen[0] - therm_cen[0]
    dY = rgb_cen[1] - therm_cen[1]
    dif_mag = np.sqrt(dX ** 2 + dY ** 2)
    angle_rad = np.arctan2(dY, dX)  # atan2(dY, dX) gives the angle in the standard coordinate system
    angle_deg = np.degrees(angle_rad)
    return dif_mag, angle_deg


rgb_cen = (220, 260)
device_list = []
c2 = []
local_directory = '/home/canyon/S3bucket/'
doc_path = '/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt'
with open(doc_path, 'r') as file:
    for line in file:
        device_list.append(line.split()[0])
        c2.append(float(line.split()[1]))

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
bucket_name = 'kcam-calibration-data'

# Example usage:
fig, ax = plt.subplots(1, 1)
vectors = np.genfromtxt("Test_vec_integrated.csv", delimiter=",", skip_header=1, dtype='<U25')
label = ''
bin_ct = 15
integ_outside = new_outside = 0
xs = []
ys = []
new_xs = []
new_ys = []

for mag, ang, id, is_rma, bad_rois in vectors:
    # print(mag, ang, id, is_rma)
    if is_rma == 'True':
        vector_plot(float(mag), float(ang), 'red', 9999)
    else:
        vector_plot(float(mag), float(ang))
c = np.asarray([int(i) for i in vectors[:, 4]])
# print(min(c),max(c))
# for i in range(len(xs)):
#     ax.scatter(xs[i], ys[i], s=90, c=c[i], cmap=cmap, norm=norm, zorder=-c[i])
ax.set_aspect('equal')
# Sort data based on c
sorted_indices = np.argsort(c)
sorted_xs = np.asarray(xs)[sorted_indices]
sorted_ys = np.asarray(ys)[sorted_indices]
sorted_c = c[sorted_indices]

# Calculate indices for partitions
third = len(c) // 3
two_thirds = 2 * third

# Partition data
xs_lowest_third = sorted_xs[:third]
ys_lowest_third = sorted_ys[:third]

xs_lowest_two_thirds = sorted_xs[:two_thirds]
ys_lowest_two_thirds = sorted_ys[:two_thirds]

xs_all = sorted_xs
ys_all = sorted_ys


# Fit ellipses
def fit_ellipse(xs, ys, ax, color, label, scale):
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    cov_matrix = np.cov(xs, ys)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ellipse = patches.Ellipse((mean_x, mean_y), 2 * np.sqrt(5 * scale * eigenvalues[0]),
                              2 * np.sqrt(5 * scale * eigenvalues[1]),
                              angle=angle, edgecolor=color, facecolor='none', label=label)
    ax.add_patch(ellipse)


# Plotting
# fig, ax = plt.subplots()

# # Plot points
# ax.scatter(xs_all, ys_all, c=c, cmap='viridis', label='All Points')

# Fit ellipses for each group
fit_ellipse(xs_lowest_third, ys_lowest_third, ax, 'green', 'Lowest 1/3', 1)
fit_ellipse(xs_lowest_two_thirds, ys_lowest_two_thirds, ax, 'green', 'Lowest 2/3', 2)
fit_ellipse(xs_all, ys_all, ax, 'green', 'All Points', 3)

# cx, cy = int(statistics.mean(xs)),  int(statistics.mean(ys))
# radius = 12
# circle = patches.Circle((cx, cy), radius, edgecolor='green', facecolor='none', linewidth=2, zorder=99999)
# ax.add_patch(circle)

# for i in range(len(xs)):
#     x = xs[i]
#     y = ys[i]
#     distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
#     if distance > radius:
#         integ_outside +=1

# print((max(c2)))
norm = mcolors.Normalize(vmin=1, vmax=max(c2))
cmap = plt.cm.get_cmap('inferno')
sm = ScalarMappable(cmap=cmap, norm=norm)
# ticks = [10**i for i in range(int(np.floor(np.log10(0))), int(np.ceil(np.log10(len(new_xs)))) + 1)]
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Values')
# #Show ticks at powers of 10
# cbar.set_ticks(ticks)
# cbar.set_ticklabels(ticks)

colors = ['orange', 'yellow', 'green', 'cyan', 'indigo', 'magenta', 'brown']
cidx = 0
for dev_id in device_list:
    label = dev_id
    # print(dev_id)
    try:
        vector = get_vector(dev_id)
    except:
        continue
    if vector:
        # vector_plot(*vector, colors[cidx%len(colors)], 999)
        vector_plot(*vector, 'yellow', 9999)
        cidx += 1
for i in range(len(new_xs)):
    ax.scatter(new_xs[i], new_ys[i], s=90, c=c2[i], cmap=cmap, norm=norm, zorder=-c2[i])

# for i in range(len(new_xs)):
#     x = new_xs[i]
#     y = new_ys[i]
#     distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
#     if distance > radius:
#         new_outside +=1

# print("Integ outside", integ_outside/len(xs))
# print("New outside", new_outside/len(new_xs))

ax.set_aspect('equal')
# Set the limits of the plot to be slightly larger than the vector
# magnitude = np.max(vectors[:,0])
ax.set_ylim(min(ys) - 15, max(ys) + 15)
ax.set_xlim(min(xs) - 15, max(xs) + 15)
# plt.legend()
plt.show()
