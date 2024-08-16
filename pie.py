import matplotlib.pyplot as plt
import numpy as np

# heads
labels = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
]
# sizes = [166,53,46,27,38,59,98,36,50,10,9,14,5,2,2,3,0,2,0,0,1,2,0,1]
sizes = [
    186,
    64,
    52,
    28,
    42,
    62,
    108,
    42,
    67,
    11,
    13,
    34,
    5,
    3,
    4,
    4,
    0,
    2,
    0,
    0,
    1,
    2,
    0,
    1,
]  # w/ winegard
# hubs
# labels = [0,1,2,3,4,5,6,7,8,9,10,11,12]
# sizes = [6,4,0,6,38,14,23,34,15,18,25,8,6]
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts = ax.pie(sizes, labels=labels, startangle=90, counterclock=False)

# Draw a circle at the center with white fill
centre_circle = plt.Circle((0, 0), 0.70, fc="white")
fig.gca().add_artist(centre_circle)

# Adding radial lines to divide the pie chart into 10 sections
for theta in np.linspace(0, 2 * np.pi, 10, endpoint=False):
    x = [0, np.cos(theta - 0.5 * np.pi)]
    y = [0, np.sin(theta - 0.5 * np.pi)]
    ax.plot(x, y, color="black", linestyle="-", linewidth=1)

ax.axis("equal")
plt.show()
