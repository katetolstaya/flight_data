import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import math
import pylab as pl
import scipy as sp
import numpy as np
import serial as ser
from random import *


# Code assumes quartic splines are being used

def Bspline4(CP, nfunctionpoints, tk, maxtk):
    """
    P0 through P11 are the (x,y,z) control points that define the quartic B-spline.
    The spline is clamped so the endpoints are coincident with the first and last control points.
    nPoints is the number of points to compute to visualize/plot this curve segment.
    """
    # Convert the points to numpy so that we can do array multiplication
    # max

    B4functions = np.zeros((nfunctionpoints, len(CP)))
    dB4functions = np.zeros((nfunctionpoints, len(CP)))
    tau = np.zeros(nfunctionpoints)
    for i in range(nfunctionpoints):
        tau[i] = maxtk * float(i) / nfunctionpoints
        # B4 functions are dimensioned as [parameter value,Column ID]
        AA, BB = B4eval(tau[i], tk)
        B4functions[i, int(np.floor(tau[i])):int(np.floor(tau[i] + 5))] = AA
        dB4functions[i, int(np.floor(tau[i])):int(np.floor(tau[i] + 5))] = BB
        B4 = B4functions.transpose()
    path = np.matmul(B4functions, CP)
    return path, B4, tau


# B4eval function only accepts clamped uniform knot vectors (tk is the knot vector.
# Function returns a [1 x 5] vector of basis function values for the parameter t
def B4eval(t, tk):
    if tk[0] == 0:
        if t >= np.amax(tk):
            t = 0.999999 * t  # Ensure index does not spill outside of spline domain
    else:
        if t >= np.amax(tk) - 4:  # in this case knot vector should start with -4
            t = 0.999999 * t
    knot = np.floor(t)
    if tk[0] == 0:
        if knot == 0:
            knotindex = int(np.amin(tk.ravel().nonzero()) - 1)
        else:
            knotindex = int(np.amin(tk.ravel().nonzero()) - 1 + knot)
    else:
        knotindex = int(np.amin((tk == knot).nonzero()))
    # Determine basis function values using Morken and Lyche recursive formula
    Blinear = np.array([(tk[knotindex + 1] - t) / (tk[knotindex + 1] - tk[knotindex]),
                        (t - tk[knotindex]) / (tk[knotindex + 1] - tk[knotindex])])
    Bquadmat = np.array([[(tk[knotindex + 1] - t) / (tk[knotindex + 1] - tk[knotindex - 1]),
                          (t - tk[knotindex - 1]) / (tk[knotindex + 1] - tk[knotindex - 1]), 0],
                         [0, (tk[knotindex + 2] - t) / (tk[knotindex + 2] - tk[knotindex]),
                          (t - tk[knotindex]) / (tk[knotindex + 2] - tk[knotindex])]])
    Bquadratic = np.matmul(Blinear, Bquadmat)
    Bcubicmat = np.array([[(tk[knotindex + 1] - t) / (tk[knotindex + 1] - tk[knotindex - 2]),
                           (t - tk[knotindex - 2]) / (tk[knotindex + 1] - tk[knotindex - 2]), 0, 0],
                          [0, (tk[knotindex + 2] - t) / (tk[knotindex + 2] - tk[knotindex - 1]),
                           (t - tk[knotindex - 1]) / (tk[knotindex + 2] - tk[knotindex - 1]), 0],
                          [0, 0, (tk[knotindex + 3] - t) / (tk[knotindex + 3] - tk[knotindex]),
                           (t - tk[knotindex]) / (tk[knotindex + 3] - tk[knotindex])]])
    Bcubic = np.matmul(Bquadratic, Bcubicmat)
    Bquarticmat = np.array([[(tk[knotindex + 1] - t) / (tk[knotindex + 1] - tk[knotindex - 3]),
                             (t - tk[knotindex - 3]) / (tk[knotindex + 1] - tk[knotindex - 3]), 0, 0, 0],
                            [0, (tk[knotindex + 2] - t) / (tk[knotindex + 2] - tk[knotindex - 2]),
                             (t - tk[knotindex - 2]) / (tk[knotindex + 2] - tk[knotindex - 2]), 0, 0],
                            [0, 0, (tk[knotindex + 3] - t) / (tk[knotindex + 3] - tk[knotindex - 1]),
                             (t - tk[knotindex - 1]) / (tk[knotindex + 3] - tk[knotindex - 1]), 0],
                            [0, 0, 0, (tk[knotindex + 4] - t) / (tk[knotindex + 4] - tk[knotindex]),
                             (t - tk[knotindex]) / (tk[knotindex + 4] - tk[knotindex])]])
    Bquartic = np.matmul(Bcubic, Bquarticmat)
    # Evaluate first derivative of basis functions at the same paramter value t..:
    dmat = np.array(
        [[-1. / (tk[knotindex + 1] - tk[knotindex + 1 - 4]), 1. / (tk[knotindex + 1] - tk[knotindex + 1 - 4]), 0, 0, 0],
         [0, -1. / (tk[knotindex + 2] - tk[knotindex + 2 - 4]), 1. / (tk[knotindex + 2] - tk[knotindex + 2 - 4]), 0, 0],
         [0, 0, -1. / (tk[knotindex + 3] - tk[knotindex + 3 - 4]), 1. / (tk[knotindex + 3] - tk[knotindex + 3 - 4]), 0],
         [0, 0, 0, -1. / (tk[knotindex + 4] - tk[knotindex + 4 - 4]),
          1. / (tk[knotindex + 4] - tk[knotindex + 4 - 4])]])

    dBquartic = 4 * np.matmul(Bcubic, dmat)

    return Bquartic, dBquartic


def Greville(CP, tk, B4, tau):  # Calculate tkstar as defined by Lutterkort
    tkstar = np.zeros((len(tk) - 4 - 1))
    num_CP = len(CP)
    dimensions = np.shape(CP)
    num_CP = dimensions[0]
    spatial_dimension = dimensions[1]
    # print("lenk-5 ",len(tk)-4-1)
    for j in range(0, len(tk) - 4 - 1):  # 4 because splines are quartic....
        tkstar[j] = 0
        for i in range(j + 1, j + 5):  # first cycle is j=1,2,3,4,5 (5 not used because of how Python runs loops)
            tkstar[j] = float(tkstar[j] + float(tk[i] / 4))
    print("tkstar ", tkstar)
    # Determine basis function values at Greville Abscissae
    B4i_tstarj = np.zeros((num_CP, num_CP))
    print("B4i_tstarjIC ", B4i_tstarj)
    for i in range(0, num_CP):  # Greville Abscissae (one for each CP)
        for j in range(0, num_CP):  # Basis function values at each Greville Abscissa
            if j == num_CP - 1:
                tkstar[j] = 0.99999999999999 * tkstar[
                    j]  # need to make sure interval is open on the terminal side of the spline: [0,maxtk)
            AAstar, BBstar = B4eval(tkstar[j],
                                    tk)  # rows are for each abscissa, columns are basis functions values at the associated abscissa
            # B4functions[i, int(np.floor(t[i])):int(np.floor(t[i] + 5))] = AA
            B4i_tstarj[int(np.floor(tkstar[j])):int(np.floor(tkstar[j] + 5)), j] = AAstar
    print("B4i_tstarj ", B4i_tstarj)
    # Compute first and second weighted differences of the control points
    # (Lutterkort uses the nomenclature b to represent control points...)
    dCP_prime = np.zeros((num_CP - 1, spatial_dimension))
    for i in range(1, num_CP):
        for j in range(spatial_dimension):
            dCP_prime[i - 1, j] = (CP[i][j] - CP[i - 1][j]) / (tkstar[i] - tkstar[i - 1])  # first weighted differences
    print("DCP_prime ", dCP_prime)
    dCP_prime2 = np.zeros((num_CP - 2, spatial_dimension))
    delta_minus = np.zeros((num_CP - 2, spatial_dimension))
    delta_plus = np.zeros((num_CP - 2, spatial_dimension))
    for i in range(num_CP - 2):
        for j in range(spatial_dimension):
            dCP_prime2[i, j] = dCP_prime[i + 1][j] - dCP_prime[i,][j]  # second weighted difference
            if dCP_prime2[i][j] < 0:
                delta_minus[i, j] = dCP_prime2[i][j]
            if dCP_prime2[i][j] > 0:
                delta_plus[i, j] = dCP_prime2[i][j]
            # delta_plus[i,j] = max([0],dCP_prime2[i][j])
    # print("delta_plus: ", delta_plus)
    print("delta_minus: ", delta_minus)
    print("dCP_prime2: ", dCP_prime2)
    # Use Greville abscissae to plot control polygon  bounds (these are indexed with respect to indices in the path variable
    gva = np.zeros(shape=(num_CP))
    gva[0] = 0
    gva[num_CP - 1] = len(tau) - 1
    for j in range(1, num_CP - 1):
        gva[j] = int(np.amin(np.where(tau > tkstar[j])))
    print("gva: ", gva)
    #
    klumatrix = np.zeros((num_CP, num_CP - 1))
    print("gva(5): ", int(gva[10]))
    print("gva(6): ", int(gva[11]))
    print("B4sum: ", sum(i > 0 for i in B4[2, int(gva[10]):int(gva[11])]))
    print("shape: ", np.shape(B4))
    # print("junk: ",np.transpose(B4[3,int(gva[5]):int(gva[5+1])]))
    for i in range(num_CP):
        for j in range(num_CP - 1):
            count = sum(ii > 0 for ii in (B4[i, int(gva[j]):int(gva[j + 1])]))
            if count > 0:
                any = 1
                count = 0
            klumatrix[i, j] = any
            any = 0  # (B4[i,gva(j):gva(j+1)]>0)
    print("klumatrix: ", klumatrix)
    k_lower = np.zeros((num_CP - 1))
    k_upper = np.zeros((num_CP - 1))
    for j in range(num_CP - 1):
        k_lower[j] = np.min(np.where(klumatrix[:, j] == 1))
        k_upper[j] = np.max(np.where(klumatrix[:, j] == 1))
    print("k_lower: ", k_lower)
    print("k_upper: ", k_upper)
    # find Beta_k_i function values at tkstar locations
    Beta_at_tkstar_k_calc = np.zeros(shape=(num_CP - 1, num_CP - 1))
    for k in range(1, num_CP - 1):
        Beta_at_tkstar_k_calc[k, int(k_lower[k]):int(k_upper[k])] = 0.0
        for i in range(int(k_lower[k]), int(k_upper[k])):
            if i <= k:
                for j in range(int(k_lower[k]), i):
                    Beta_at_tkstar_k_calc[k, i] = Beta_at_tkstar_k_calc[k, i] + (tkstar[i] - tkstar[j]) * B4i_tstarj[
                        j, k]
            if i > k:
                for j in range(i, int(k_upper[k])):
                    Beta_at_tkstar_k_calc[k, i] = Beta_at_tkstar_k_calc[k, i] + (tkstar[j] - tkstar[i]) * B4i_tstarj[
                        j, k]
    print("Beta_at_tkstar_k_calc: ", Beta_at_tkstar_k_calc)
    print("shape: ", np.shape(Beta_at_tkstar_k_calc))
    # lose first row and column of zeros...
    Beta_at_tkstar_k = np.zeros(shape=(num_CP - 2, num_CP - 2))
    print("Beta_at_tkstar_k: ", Beta_at_tkstar_k)
    Beta_at_tkstar_k = Beta_at_tkstar_k_calc[1:num_CP - 1, 1:num_CP - 1]
    print("Beta_at_tkstar_k: ", Beta_at_tkstar_k)
    print("shape: ", np.shape(Beta_at_tkstar_k))
    print("shape_delta_minus: ", np.shape(delta_minus))
    print("shape_delta_plus: ", np.shape(delta_plus))
    # Compute bounding envelope offsets from the control points %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    xxx = np.squeeze([Beta_at_tkstar_k.dot(delta_minus)])
    yyy = np.squeeze([Beta_at_tkstar_k.dot(delta_plus)])
    zzz = np.zeros((1, spatial_dimension))
    offset_lowerCP = np.vstack((zzz, xxx))
    offset_lowerCP = np.vstack((offset_lowerCP, zzz))
    offset_upperCP = np.vstack((zzz, yyy))
    offset_upperCP = np.vstack((offset_upperCP, zzz))
    CP_minus = CP + offset_lowerCP
    CP_plus = CP + offset_upperCP
    print("CP_minus: ", CP_minus)
    print("CP_plus: ", CP_plus)
    return (CP_minus, CP_plus)


#####################################################################

# Main Program

def main():
    # Define some random control points with which to define a spline
    numCP = 12
    nfunctionpoints = 200

    CP = np.zeros(shape=(numCP, 3))
    # Set points for debugging
    '''
    CP = [[0.0000e+000,  0.0000e+000,  0.0000e+000],
      [7.5719e+001,  1.1039e+002,  6.1945e+000],
      [2.0745e+002,  2.2078e+002,  7.3384e+001],
      [1.2975e+002,  3.3116e+002, 1.0116e+002],
      [2.5735e+002,  4.4155e+002,  9.0859e+001],
      [5.9660e+001,  5.5194e+002,  2.0811e+002],
      [2.2981e+002,  6.6233e+002,  1.7892e+002],
      [2.5013e+002,  7.7271e+002,  7.9101e+001],
      [2.5920e+002,  8.8310e+002,  1.6877e+002],
      [2.9602e+002,  9.9349e+002,  1.1527e+002],
      [1.4954e+001,  1.1039e+003,  7.6059e+001],
      [2.5631e+002,  1.2143e+003,  6.6500e+001]]
    '''
    for i in range(numCP):
        CP[i, 0] = 20 * i  # Evenly space x coordinates by 2
        CP[i, 1] = 80 * random() + 4 * i  # Randomly assign y and z coordinates
        CP[i, 2] = 80 * random() + 5 * i

    # Build knot vector tk
    maxtk = len(CP) - 4
    s = (1, 4)
    tk = np.zeros(s)
    tkmiddle = np.arange(maxtk + 1)
    tkend = maxtk * np.ones(s)
    tk = np.append(tk, tkmiddle)
    tk = np.append(tk, tkend)

    # Define Path
    path, B4, tau = Bspline4(CP, nfunctionpoints, tk, maxtk)
    print('path shape')
    print(path)

    # Get Greville Abscissae
    CP_minus, CP_plus = Greville(CP, tk, B4, tau)

    # Compose bounds in 3D for plotting (for each x (easting))
    CP_ne_minus_z_minus = CP_minus  # lower bound in y (northing) and lower bound in z (altitude)
    CP_ne_minus_z_plus = np.zeros((np.shape(CP)))  # Hard-wired for 3D control points
    CP_ne_plus_z_minus = np.zeros((np.shape(CP)))  # Hard-wired for 3D control points
    for i in range(len(CP)):
        CP_ne_minus_z_plus[i, :] = [CP_minus[i, 0], CP_minus[i, 1],
                                    CP_plus[i, 2]]  # lower bound in y (northing)and upper bound in z (altitude)
        CP_ne_plus_z_minus[i, :] = [CP_plus[i, 0], CP_plus[i, 1],
                                    CP_minus[i, 2]]  # upper bound in y (northing)and lower bound in z (altitude)
    CP_ne_plus_z_plus = CP_plus  # upper bound in y (northing)and upper bound in z (altitude)

    # Parse bounds by tkstar (build rectangles around
    bound1 = np.zeros((5, 3, len(CP)))
    # print("bound",bound1)
    boundj = np.zeros((5, 3))
    appended = ['']
    for j in range(1, len(CP) - 1):
        boundj = np.squeeze([[CP_ne_minus_z_minus[j, 0:3]], [CP_ne_plus_z_minus[j, 0:3]], [CP_ne_plus_z_plus[j, 0:3]],
                             [CP_ne_minus_z_plus[j, 0:3]], [CP_ne_minus_z_minus[j, 0:3]]])
        # print("bound1[:,:,j)", boundj)
        bound1[:, :, j] = boundj

    # Plot results
    symmetric_plot_limit = 200

    # pl.hold(True)
    ax1 = a3.Axes3D(pl.figure(1))
    ax1.set_aspect("equal")
    ax1.set_xlim(-symmetric_plot_limit * 0, symmetric_plot_limit)
    ax1.set_ylim(-symmetric_plot_limit * 0, symmetric_plot_limit)
    ax1.set_zlim(-symmetric_plot_limit * 0, symmetric_plot_limit)
    ax1.set_xlabel('East - m')
    ax1.set_ylabel('North - m')
    ax1.set_zlabel('Up - m')
    # Convert the path points into x, y and z arrays and plot
    x, y, z = zip(*path)
    ax1.plot(y, x, z, c='k', linestyle='-', linewidth=1)
    pl.grid(True)
    #
    # Plot the control points
    print("CP", CP)
    dim_bound = np.shape(bound1)
    print("dimbound1", dim_bound)
    px, py, pz = zip(*CP)
    # a3.Axes3D.plot(px, py, pz, 'or')
    ax1.scatter(py, px, pz, c='r', marker='o', s=30)
    ax1.plot(py, px, pz, c='b', linestyle=':', linewidth=2)
    # Plot bounds as line segments from start to end of curve/spline
    for k in range(dim_bound[0]):
        for n in range(dim_bound[2]):
            ax1.scatter(bound1[k][1][n], bound1[k][0][n], bound1[k][2][n], c='k', marker='p', s=20)

    # Plot rectangle around Greville bounds at each Greville Abscissa
    for n in range(dim_bound[2]):
        ax1.plot(bound1[0:dim_bound[0], 1, n], bound1[0:dim_bound[0], 0, n], bound1[0:dim_bound[0], 2, n], c='m',
                 linestyle='-', linewidth=1.5)
    #
    # Plot the bounds as line segments un magenta
    CP_ne_minus_z_minus_x, CP_ne_minus_z_minus_y, CP_ne_minus_z_minus_z = zip(*CP_ne_minus_z_minus)
    CP_ne_plus_z_minus_x, CP_ne_plus_z_minus_y, CP_ne_plus_z_minus_z = zip(*CP_ne_plus_z_minus)
    CP_ne_minus_z_plus_x, CP_ne_minus_z_plus_y, CP_ne_minus_z_plus_z = zip(*CP_ne_minus_z_plus)
    CP_ne_plus_z_plus_x, CP_ne_plus_z_plus_y, CP_ne_plus_z_plus_z = zip(*CP_ne_plus_z_plus)
    #
    ax1.scatter(CP_ne_minus_z_minus_y, CP_ne_minus_z_minus_x, CP_ne_minus_z_minus_z, c='g', marker='s', s=5)
    ax1.plot(CP_ne_minus_z_minus_y, CP_ne_minus_z_minus_x, CP_ne_minus_z_minus_z, c='m', linestyle=':', linewidth=1)
    #
    ax1.scatter(CP_ne_plus_z_minus_y, CP_ne_plus_z_minus_x, CP_ne_plus_z_minus_z, c='g', marker='s', s=5)
    ax1.plot(CP_ne_plus_z_minus_y, CP_ne_plus_z_minus_x, CP_ne_plus_z_minus_z, c='m', linestyle=':', linewidth=1)
    # CP_ne_minus_z_plus_
    ax1.scatter(CP_ne_minus_z_plus_y, CP_ne_minus_z_plus_x, CP_ne_minus_z_plus_z, c='g', marker='s', s=5)
    ax1.plot(CP_ne_minus_z_plus_y, CP_ne_minus_z_plus_x, CP_ne_minus_z_plus_z, c='m', linestyle=':', linewidth=1)
    #
    ax1.scatter(CP_ne_plus_z_plus_y, CP_ne_plus_z_plus_x, CP_ne_plus_z_plus_z, c='g', marker='s', s=5)
    ax1.plot(CP_ne_plus_z_plus_y, CP_ne_plus_z_plus_x, CP_ne_plus_z_plus_z, c='m', linestyle=':', linewidth=1)
    pl.show()


if __name__ == "__main__":
    main()
