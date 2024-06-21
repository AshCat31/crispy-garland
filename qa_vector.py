import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches
from matplotlib.cm import ScalarMappable
import numpy as np
import statistics
import boto3
import cv2
import os

def vector_plot(magnitude, angle_degrees, color='black', zorder=10):
    angle_radians = np.deg2rad(-angle_degrees+270)
    x_component = magnitude * np.cos(angle_radians)
    y_component = magnitude * np.sin(angle_radians)
    # ax.arrow(0, 0, x_component, y_component, head_width=1, color=color)
    if color == 'yellow':
        ax.plot(x_component, y_component, 'o', color=color, markersize=12, zorder=zorder)
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
        mask=np.load(os.path.join(local_directory, device_id, f'mapped_mask_matrix_hydra_{device_id}.npy'))
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

    # get vector
    dX= rgb_cen[0]-therm_cen[0]
    dY= rgb_cen[1]-therm_cen[1]
    dif_mag=np.sqrt(dX**2 + dY**2)
    angle_rad = np.arctan2(dY, dX)  # atan2(dY, dX) gives the angle in the standard coordinate system
    angle_deg = np.degrees(angle_rad)
    return dif_mag, angle_deg


rgb_cen=(220,260)
device_list = []
local_directory='/home/canyon/S3bucket/'
doc_path = '/home/canyon/Test_Equipment/QA_ids.txt'
with open(doc_path, 'r') as file:
    for line in file:
        device_list.append(line.split()[0])

cred = boto3.Session().get_credentials()
ACCESS_KEY = cred.access_key 
SECRET_KEY = cred.secret_key 
SESSION_TOKEN = cred.token 
global s3client
s3client = boto3.client('s3',
                        aws_access_key_id = ACCESS_KEY,
                        aws_secret_access_key = SECRET_KEY,
                        aws_session_token = SESSION_TOKEN,
                        )
bucket_name = 'kcam-calibration-data'

# Example usage:
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
    # print(mag, ang, id, is_rma)
    if is_rma=='True':
        vector_plot(float(mag), float(ang), 'red', 9999)
    else:
        vector_plot(float(mag), float(ang))
c = np.asarray([int(i) for i in vectors[:,4]])
norm = mcolors.LogNorm(vmin=min(c)+1, vmax=max(c))
# print(min(c),max(c))
cmap = plt.cm.get_cmap('twilight')
for i in range(len(xs)):
    ax.scatter(xs[i], ys[i], s=90, c=c[i], cmap=cmap, norm=norm, zorder=-c[i])
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Values')
#Show ticks at powers of 10
ticks = [10**i for i in range(int(np.floor(np.log10(min(c) + 1))), int(np.ceil(np.log10(max(c)))) + 1)]
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticks)

ax.set_aspect('equal')

cx, cy = int(statistics.mean(xs)),  int(statistics.mean(ys))
radius = 12
circle = patches.Circle((cx, cy), radius, edgecolor='green', facecolor='none', linewidth=2, zorder=99999)
ax.add_patch(circle)

for i in range(len(xs)):
    x = xs[i]
    y = ys[i]
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    if distance > radius:
        integ_outside +=1

colors = ['orange','yellow','green','cyan','indigo','magenta', 'brown']
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
        cidx+=1

for i in range(len(new_xs)):
    x = new_xs[i]
    y = new_ys[i]
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    if distance > radius:
        new_outside +=1

print("Integ outside", integ_outside/len(xs))
print("New outside", new_outside/len(new_xs))

ax.set_aspect('equal')
# Set the limits of the plot to be slightly larger than the vector
# magnitude = np.max(vectors[:,0])
ax.set_ylim(min(ys)-15,max(ys)+15)
ax.set_xlim(min(xs)-15,max(xs)+15)
# plt.legend()
plt.show()