import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.pyplot import gca


rc('text', usetex=True)
font = {'family' : 'Times New Roman', 'weight' : 'bold', 'size'   : 14}
rc('font', **font)

path = "logs/grid19_restore2.txt"

d = np.loadtxt(path, delimiter="\t", skiprows=1)

inds = d[:, 0]
planner_cost = d[:, 1]
expert_cost = d[:, 2]
diff = d[:, 3]

obj = planner_cost - expert_cost

N =  20
filt_obj = np.convolve(obj, np.ones((N,))/N, mode='same')

print(inds)

fig, ax = plt.subplots()
ax.plot(inds, filt_obj)
plt.xlabel('Iteration Index')
plt.ylabel("Objective $L(J_a)$")
#plt.title("IRL Objective")
a = gca()
a.set_xticklabels(a.get_xticks(), font)
a.set_yticklabels(a.get_yticks(), font)
plt.tight_layout()
plt.show()

# plt.ylim(-1.0 * problem.r_max, 1.0 * problem.r_max)
# plt.xlim(-1.0 * problem.r_max, 1.0 * problem.r_max)
# a = gca()
# a.set_xticklabels(a.get_xticks(), font)
# a.set_yticklabels(a.get_yticks(), font)
# plt.title('GNN Controller')

