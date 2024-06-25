import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches
from matplotlib.cm import ScalarMappable
import numpy as np
import statistics
import os

from s3_setup import setup_s3


def vector_plot(magnitude, angle_degrees, color='black', zorder=10):
    angle_radians = np.deg2rad(-angle_degrees+270)
    x_component = magnitude * np.cos(angle_radians)
    y_component = magnitude * np.sin(angle_radians)
    if color == 'yellow':
        if ellipse_a.contains_point(ax.transData.transform((x_component, y_component))):
            color = 'green'
        elif ellipse_c.contains_point(ax.transData.transform((x_component, y_component))) and not ellipse_b.contains_point(ax.transData.transform((x_component, y_component))):
            color = 'orange'
        results_dict[color] +=1
    if color == 'black':
        xs.append(x_component)
        ys.append(y_component)
    elif color != 'red':
        new_xs.append(x_component)
        new_ys.append(y_component)
def ensure_path_exists(key):
    path_components = key.split('/')
    recursive_create_directories(path_components, -2)
def recursive_create_directories(path_components, index):
    if index < 0:
        return  # Base case: reached the beginning of the path components
    current_path = '/'.join(path_components[:index+1])
    if not os.path.exists(current_path):
        os.makedirs(current_path)
        print(f"Created directory: {current_path}")
    recursive_create_directories(path_components, index - 1)
def get_vector(device_id):
    try:
        mask=np.load(os.path.join(local_directory, device_id, f'mapped_mask_matrix_hydra_{device_id}.npy'))
    except FileNotFoundError:
        try:
            key = f'{device_id}/calculated_transformations2/{device_id}/mapped_mask_matrix_hydra_{device_id}.npy'
            ensure_path_exists(key)
            s3client.download_file(Bucket=bucket_name,
                        Key=key,
                        Filename=os.path.join(local_directory, device_id, f'mapped_mask_matrix_hydra_{device_id}.npy')) 
        except FileNotFoundError:
            try:
                s3client.download_file(Bucket=bucket_name,
                            Key=f'{device_id}/calculated_transformations/{device_id}/mapped_mask_matrix_hydra_{device_id}.npy',
                            Filename=os.path.join(local_directory, device_id, f'mapped_mask_matrix_hydra_{device_id}.npy')) 
            except FileNotFoundError:
                print("Error:", device_id, "has no mapped mask matrix")
                return None
    try:
        mask=np.load(os.path.join(local_directory, device_id, f'mapped_mask_matrix_hydra_{device_id}.npy'))
    except FileNotFoundError:
        print("Error:", device_id, "not found")
        return None
    mask = mask.astype(np.uint8) * 255
    therm_x = statistics.mean(np.nonzero(mask)[0])
    therm_y = statistics.mean(np.nonzero(mask)[1])
    therm_cen=(therm_x,therm_y)
    dX= rgb_cen[0]-therm_cen[0]
    dY= rgb_cen[1]-therm_cen[1]
    dif_mag=np.sqrt(dX**2 + dY**2)
    angle_rad = np.arctan2(dY, dX)  # atan2(dY, dX) gives the angle in the standard coordinate system
    angle_deg = np.degrees(angle_rad)
    return dif_mag, angle_deg
rgb_cen=(220,260)
device_list = []
c2 = []
local_directory='/home/canyon/S3bucket/'
doc_path = '/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt'
with open(doc_path, 'r') as file:
    for line in file:
        device_list.append(line.split()[0])
        c2.append(float(line.split()[1]))
global s3client
s3client, bucket_name = setup_s3()
fig, ax = plt.subplots(1,1)
vectors = np.genfromtxt("Test_vec_integrated.csv", delimiter=",", skip_header=1, dtype='<U25')
label = ''
bin_ct = 15
integ_outside = new_outside = 0
xs = []
ys = []
new_xs = []
new_ys = []
for mag, ang, id, is_rma, bad_rois in vectors:
    if is_rma=='True':
        vector_plot(float(mag), float(ang), 'red', 9999)
    else:
        vector_plot(float(mag), float(ang))
c = np.asarray([int(i) for i in vectors[:,4]])
cmap = plt.cm.get_cmap('inferno')
ax.set_aspect('equal')
sorted_indices = np.argsort(c)
sorted_xs = np.asarray(xs)[sorted_indices]
sorted_ys = np.asarray(ys)[sorted_indices]
sorted_c = c[sorted_indices]
third = len(c) // 3
two_thirds = 2 * third
xs_lowest_third = sorted_xs[:third]
ys_lowest_third = sorted_ys[:third]
xs_lowest_two_thirds = sorted_xs[:two_thirds]
ys_lowest_two_thirds = sorted_ys[:two_thirds]
xs_all = sorted_xs
ys_all = sorted_ys
def fit_ellipse(xs, ys, ax, color, label, scale):
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    cov_matrix = np.cov(xs, ys)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ellipse = patches.Ellipse((mean_x, mean_y), 2 * np.sqrt(5*scale * eigenvalues[0]), 2 * np.sqrt(5*scale * eigenvalues[1]),
                      angle=angle, edgecolor=color, facecolor='none', label=label, zorder=99999)
    ax.add_patch(ellipse)
    return ellipse
ellipse_a = fit_ellipse(xs_lowest_third, ys_lowest_third, ax, 'green', 'Lowest 1/3', 1)
ellipse_b = fit_ellipse(xs_lowest_two_thirds, ys_lowest_two_thirds, ax, 'green', 'Lowest 2/3', 2)
ellipse_c = fit_ellipse(xs_all, ys_all, ax, 'green', 'All Points', 3)
results_dict = {'green':0, 'yellow':0, 'orange':0}
colors = ['orange','yellow','green','cyan','indigo','magenta', 'brown']
cidx = 0
for dev_id in device_list:
    label = dev_id
    try:
        vector = get_vector(dev_id)
    except Exception as e:
        continue
    if vector:
        vector_plot(*vector, 'yellow', 9999)
        cidx+=1
mean_x = np.mean(xs_all)
mean_y = np.mean(ys_all)
cov_matrix = np.cov(xs_all, ys_all)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
scale = 3  # adjust scale as needed
def distance_to_center(x, y):
    return np.sqrt((x - mean_x)**2 + (y - mean_y)**2)
def distance_to_boundary(x, y):
    dx = x - mean_x
    dy = y - mean_y
    rotated_dx = dx * np.cos(np.radians(angle)) + dy * np.sin(np.radians(angle))
    rotated_dy = -dx * np.sin(np.radians(angle)) + dy * np.cos(np.radians(angle))
    ellipse_eq = (rotated_dx / np.sqrt(eigenvalues[0] * scale))**2 + (rotated_dy / np.sqrt(eigenvalues[1] * scale))**2
    return np.abs(np.sqrt(1 / ellipse_eq) * scale)
distances_to_center = [distance_to_center(x, y) for x, y in zip(new_xs, new_ys)]
distances_to_boundary = [distance_to_boundary(x, y) for x, y in zip(new_xs, new_ys)]
ratios = [distances_to_center[i] / distances_to_boundary[i] for i in range(len(new_xs))]
ratio_sorted_indices = sorted(range(len(ratios)), key=lambda k: ratios[k])
c2_sorted_indices = sorted(range(len(c2)), key=lambda k: c2[k])
num_indices = len(ratio_sorted_indices)
average_positions = [(ratio_sorted_indices[i] + c2_sorted_indices[i]) / 2 for i in range(num_indices)]
combined_indices = list(zip(average_positions, range(num_indices)))
average_sorted_indices = [index for _, index in sorted(combined_indices)]
num_points = len(average_sorted_indices)
norm = mcolors.LogNorm(vmin=1, vmax=len(average_sorted_indices))
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Values')
def get_hex_color(index):
    rgba_color = cmap(norm(index))
    hex_color = mcolors.rgb2hex(rgba_color)
    return hex_color
for idx, i in enumerate(average_sorted_indices):
    ax.scatter(new_xs[i], new_ys[i], s=90, c=get_hex_color(idx), zorder=-idx)
print("Good", results_dict['green'])
print("Ok", results_dict['yellow'])
print("Bad", results_dict['orange'])
print(len(np.nonzero(c<1)[0]))

ax.set_aspect('equal')
ax.set_ylim(min(ys)-15,max(ys)+15)
ax.set_xlim(min(xs)-15,max(xs)+15)
plt.show()
