import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.pyplot import gca
from math import ceil

rc('text', usetex=True)
font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18}
rc('font', **font)

path = "test_good.txt"
d = np.loadtxt(path, delimiter="\t", skiprows=2)
good_diff = d[:, 3]

path = "test_bad.txt"
d = np.loadtxt(path, delimiter="\t", skiprows=2)
bad_diff = d[:, 3]

exps = [ 'Path Length Only', 'Learned Routing Cost']
x_pos = np.arange(len(exps))
CTEs = [np.mean(bad_diff), np.mean(good_diff)]
error = [np.std(bad_diff), np.std(good_diff)]

# Build the plot
fig, ax = plt.subplots()
pb, pg = ax.bar(x_pos, CTEs, yerr=error, align='center', capsize=10, ecolor='black')

pb.set_facecolor('lightcoral')
pg.set_facecolor('darkseagreen')

ax.set_ylabel('Avg. Min. Path Difference (km)')
ax.set_xticks(x_pos)
ax.set_xticklabels(exps)
# ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bad_good.png')
plt.show()

