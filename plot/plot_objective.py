import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.pyplot import gca
from math import ceil

rc('text', usetex=True)
font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 14}
rc('font', **font)

path = "log1000.txt"

d = np.loadtxt(path, delimiter="\t", skiprows=1)

inds = d[:, 0]
planner_cost = d[:, 1]
expert_cost = d[:, 2]
diff = d[:, 3]

step_ind = 50
min_ind = 1000
num_inds = int(np.shape(planner_cost)[0] / step_ind)

obj = planner_cost - expert_cost

N = 50

##################################################################################################
### Objective Plot
filt_obj = np.convolve(obj, np.ones((N,)) / N, mode='same')

avg_cost = np.zeros((num_inds,))
for i in range(num_inds):
    avg_cost[i] = np.sum(obj[i * step_ind:((i + 1) * step_ind)]) / step_ind

num_inds = np.shape(avg_cost)[0]
inds = np.linspace(0, num_inds * step_ind, num_inds)

fig, ax = plt.subplots()
ax.plot(inds, avg_cost)
plt.xlabel('Training Steps')
plt.ylabel("Margin $\hat{\mathcal{L}}(J_a)$")  # plt.ylabel("Objective $\mathcal{L}(J_a)$")
# plt.title("IRL Objective")
a = gca()
a.set_xticklabels(a.get_xticks(), font)
a.set_yticklabels(a.get_yticks(), font)
plt.tight_layout()
plt.savefig('margin.png')
plt.show()

################################################################################################
### % Timeout plot
freq_timeout = np.zeros((num_inds,))
planner_cost_nonzero = planner_cost > 0

for i in range(num_inds):
    freq_timeout[i] = np.sum(planner_cost_nonzero[i * step_ind:((i + 1) * step_ind)])
freq_timeout[int(min_ind / step_ind):] = freq_timeout[int(min_ind / step_ind):] / step_ind

num_inds = np.shape(avg_cost)[0]
inds = np.linspace(0, num_inds * step_ind, num_inds)

fig, ax = plt.subplots()
ax.plot(inds, 1 - freq_timeout)
plt.xlabel('Training Steps')
plt.ylabel("Fraction of Time-Outs")

plt.tight_layout()
plt.savefig('timeouts.png')
plt.show()
