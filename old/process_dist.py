import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import math
from parse_data import Flight
from sklearn.neighbors import NearestNeighbors
import numpy as np


#file_names = ['0501', '0502', '0503', '0504', '0505', '0506', '0507', '0508', '0509', '0510']
file_names = ['20160111', '20160112', '20160113']
file_names = [ '20160111']
hist_dist = np.zeros((0,2))

for fname in file_names:
    dist = pickle.load(open('flights'+fname+'_dists.pkl', 'rb'))
    hist_dist = np.vstack([hist_dist, dist])
    print(fname)

#hist_dist = hist_dist[np.all(hist_dist, axis=1)]

max_dim = 10000
xmax = 10000
ymax = 10000
xmin = 0.1
ymin = 0.0
print(hist_dist)

## dist to goal, dist to neighbors
H, xedges, yedges = np.histogram2d(hist_dist[:, 0], 1000000.0/hist_dist[:, 1], bins = 50, range=[[xmin, xmax], [ymin, ymax]])

fig = plt.figure(figsize=(7, 3))
fig.tight_layout()
ax = fig.add_subplot(132)
X, Y = np.meshgrid(xedges, yedges, indexing='ij')
ax.pcolormesh(X, Y, H)

plt.xlabel('Distance to Goal')
plt.ylabel('Distance to Neighbors')


plt.show()
