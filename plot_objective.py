import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.pyplot import gca
from math import ceil

rc('text', usetex=True)
font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 14}
rc('font', **font)

path = "logs/grid19_restore2.txt"

d = np.loadtxt(path, delimiter="\t", skiprows=1)

inds = d[:, 0]
planner_cost = d[:, 1]
expert_cost = d[:, 2]
diff = d[:, 3]

obj = planner_cost - expert_cost

N = 20

# filt_obj = np.convolve(obj, np.ones((N,))/N, mode='same')
# fig, ax = plt.subplots()
# ax.plot(inds, filt_obj)
# plt.xlabel('Iteration Index')
# plt.ylabel("Objective $L(J_a)$") # plt.ylabel("Objective $\mathcal{L}(J_a)$")
# #plt.title("IRL Objective")
# a = gca()
# a.set_xticklabels(a.get_xticks(), font)
# a.set_yticklabels(a.get_yticks(), font)
# plt.tight_layout()
# plt.show()

planner_cost_nonzero = planner_cost > 0
step_ind = 50
min_ind = 5000
num_inds = int(np.shape(planner_cost)[0] / step_ind)
freq_timeout = np.zeros((num_inds,))

for i in range(num_inds):
    freq_timeout[i] = np.sum(planner_cost_nonzero[i * step_ind:((i + 1) * step_ind)])

freq_timeout[int(min_ind / step_ind):] = freq_timeout[int(min_ind / step_ind):] / step_ind

N = 20
freq_timeout = np.convolve(freq_timeout, np.ones((N,)) / N, mode='valid')
num_inds = np.shape(freq_timeout)[0]
inds = np.linspace(0, num_inds * step_ind, num_inds)

fig, ax = plt.subplots()
ax.plot(inds, 1 - freq_timeout)
plt.xlabel('Iteration Index')
plt.ylabel("Fraction of Plans Timed Out")

plt.tight_layout()
plt.show()

# plt.ylim(-1.0 * problem.r_max, 1.0 * problem.r_max)
# plt.xlim(-1.0 * problem.r_max, 1.0 * problem.r_max)
# a = gca()
# a.set_xticklabels(a.get_xticks(), font)
# a.set_yticklabels(a.get_yticks(), font)
# plt.title('GNN Controller')
