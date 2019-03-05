# n_path_pts = path.shape[0]
#
# # Build knot vector tk
# maxtk = n_path_pts - 4
# s = (1, 4)
# tk = np.zeros(s)
# tkmiddle = np.arange(maxtk + 1)
# tkend = maxtk * np.ones(s)
# tk = np.append(tk, tkmiddle)
# tk = np.append(tk, tkend)
#
# ts = np.linspace(start=path[0, 4], stop=path[-1, 4], num=n_ts)
# smoothed_path, B4, tau = Bspline4(path[:, 0:3], n_ts, tk, maxtk)  # interpolate in XYZ
#
# xs = smoothed_path[:, 0].reshape(-1, 1)
# ys = smoothed_path[:, 1].reshape(-1, 1)
# zs = smoothed_path[:, 2].reshape(-1, 1)
# bs = np.arctan2(ys[1:] - ys[:-1], xs[1:] - xs[:-1])
# bs = np.append(bs, [bs[-1]], axis=0)
# ts = ts.reshape(-1, 1)
#
# smoothed_path = np.stack((xs, ys, zs, bs, ts), axis=1).reshape(-1, 5)
# return smoothed_path