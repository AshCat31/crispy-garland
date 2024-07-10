import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse


def vector_plot(magnitude, angle_degrees, color='black', zorder=10):
    angle_radians = np.deg2rad(-angle_degrees + 270)
    x_component = magnitude * np.cos(angle_radians)
    y_component = magnitude * np.sin(angle_radians)
    xs.append(x_component)
    ys.append(y_component)



# Example usage:
fig, ax = plt.subplots(1, 1)
vectors = np.genfromtxt("Test_vec_integrated.csv", delimiter=",", skip_header=1, dtype='<U25')
bin_ct = 15
xs = []
ys = []
for mag, ang, id, is_rma, bad_rois in vectors:
    if is_rma == 'True':
        vector_plot(float(mag), float(ang), 'red', 9999)
    else:
        vector_plot(float(mag), float(ang))

print(len(xs), len(vectors[:, 3]))
c = np.asarray([int(i) for i in vectors[:, 4]])
norm = mcolors.LogNorm(vmin=min(c) + 1, vmax=max(c))
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
fit_ellipse(xs_lowest_third, ys_lowest_third, ax, 'red', 'Lowest 1/3', 1)
fit_ellipse(xs_lowest_two_thirds, ys_lowest_two_thirds, ax, 'blue', 'Lowest 2/3', 2)
fit_ellipse(xs_all, ys_all, ax, 'black', 'All Points', 3)

plt.show()
