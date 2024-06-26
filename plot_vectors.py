import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse
import numpy as np


def vector_plot(magnitude, angle_degrees, color='black', zorder=10):
    angle_radians = np.deg2rad(-angle_degrees + 270)
    x_component = magnitude * np.cos(angle_radians)
    y_component = magnitude * np.sin(angle_radians)
    # ax.arrow(0, 0, x_component, y_component, head_width=1, color='b', edgecolor='black')
    # ax.plot(x_component, y_component, 'o', color=color, markersize=5, zorder=zorder, c=)
    xs.append(x_component)
    ys.append(y_component)

    # ax.annotate(id, xy=(x_component, y_component), xytext=(x_component+0.1, y_component+0.1), fontsize=8, color='dimgrey')


# Example usage:
fig, ax = plt.subplots(1, 1)
vectors = np.genfromtxt("Test_vec_integrated.csv", delimiter=",", skip_header=1, dtype='<U25')
bin_ct = 15
# counts, edges, bars = ax[0].hist([float(i) for i in vectors[:,0]],bins=bin_ct,edgecolor='black')
# ax[0].set_title('Magnitude')
# ax[0].bar_label(bars)
# counts, edges, bars = ax[1].hist([float(i)+180 for i in vectors[:,1]],bins=bin_ct,edgecolor='black')
# ax[1].set_title('Angle')
# ax[1].bar_label(bars)
xs = []
ys = []
for mag, ang, id, is_rma, bad_rois in vectors:
    # print(mag, ang, id, is_rma, bad_rois)
    if is_rma == 'True':
        vector_plot(float(mag), float(ang), 'red', 9999)
    else:
        vector_plot(float(mag), float(ang))

# circle = patches.Circle(int((statistics.mean(xs))), int(statistics.mean(ys)), edgecolor='yellow', facecolor='none', linewidth=2, zorder=999)
# ax.add_patch(circle)
print(len(xs), len(vectors[:, 3]))
c = np.asarray([int(i) for i in vectors[:, 4]])
norm = mcolors.LogNorm(vmin=min(c) + 1, vmax=max(c))
# print(min(c),max(c))
cmap = plt.cm.get_cmap('twilight')
for i in range(len(xs)):
    ax.scatter(xs[i], ys[i], s=90, c=c[i], cmap=cmap, norm=norm, zorder=-c[i])
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Number of Bad ROIs')
# Show ticks at powers of 10
ticks = [10 ** i for i in range(int(np.floor(np.log10(min(c) + 1))), int(np.ceil(np.log10(max(c)))) + 1)]
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticks)

ax.set_aspect('equal')
# Set the limits of the plot to be slightly larger than the vector
# magnitude = np.max(vectors[:,0])
ax.set_ylim(min(ys) - 15, max(ys) + 15)
ax.set_xlim(min(xs) - 15, max(xs) + 15)
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
    ellipse = Ellipse((mean_x, mean_y), 2 * np.sqrt(5 * scale * eigenvalues[0]),
                      2 * np.sqrt(5 * scale * eigenvalues[1]),
                      angle=angle, edgecolor=color, facecolor='none', label=label)
    ax.add_patch(ellipse)


# Plotting
# fig, ax = plt.subplots()

# # Plot points
# ax.scatter(xs_all, ys_all, c=c, cmap='viridis', label='All Points')

# Fit ellipses for each group
fit_ellipse(xs_lowest_third, ys_lowest_third, ax, 'red', 'Lowest 1/3', 1)
fit_ellipse(xs_lowest_two_thirds, ys_lowest_two_thirds, ax, 'blue', 'Lowest 2/3', 2)
fit_ellipse(xs_all, ys_all, ax, 'black', 'All Points', 3)

plt.show()
