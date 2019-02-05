"""
Dubins path planner sample code
author Atsushi Sakai(@Atsushi_twi)
"""

import math
from math import cos, sin, atan2, floor, sqrt, acos

inf = float('inf')
pi = math.pi


def neg_pi_to_pi(theta):
    return (theta + pi) % (2 * pi) - pi


def mod2pi(theta):
    return theta - 2.0 * pi * floor(theta * 0.5 / pi)


def LSL(alpha, beta, d):
    sa = sin(alpha)
    sb = sin(beta)
    ca = cos(alpha)
    cb = cos(beta)
    c_ab = cos(alpha - beta)

    tmp0 = d + sa - sb

    mode = ['L', 'S', 'L']
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sa - sb))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = atan2((cb - ca), tmp0)
    t = mod2pi(-alpha + tmp1)
    p = sqrt(p_squared)
    q = mod2pi(beta - tmp1)
    #  print(np.rad2deg(t), p, np.rad2deg(q))

    return t, p, q, mode


def RSR(alpha, beta, d):
    sa = sin(alpha)
    sb = sin(beta)
    ca = cos(alpha)
    cb = cos(beta)
    c_ab = cos(alpha - beta)

    tmp0 = d - sa + sb
    mode = ['R', 'S', 'R']
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sb - sa))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = atan2((ca - cb), tmp0)
    t = mod2pi(alpha - tmp1)
    p = sqrt(p_squared)
    q = mod2pi(-beta + tmp1)

    return t, p, q, mode


def LSR(alpha, beta, d):
    sa = sin(alpha)
    sb = sin(beta)
    ca = cos(alpha)
    cb = cos(beta)
    c_ab = cos(alpha - beta)

    p_squared = -2 + (d * d) + (2 * c_ab) + (2 * d * (sa + sb))
    mode = ['L', 'S', 'R']
    if p_squared < 0:
        return None, None, None, mode
    p = sqrt(p_squared)
    tmp2 = atan2((-ca - cb), (d + sa + sb)) - atan2(-2.0, p)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(-mod2pi(beta) + tmp2)

    return t, p, q, mode


def RSL(alpha, beta, d):
    sa = sin(alpha)
    sb = sin(beta)
    ca = cos(alpha)
    cb = cos(beta)
    c_ab = cos(alpha - beta)

    p_squared = (d * d) - 2 + (2 * c_ab) - (2 * d * (sa + sb))
    mode = ['R', 'S', 'L']
    if p_squared < 0:
        return None, None, None, mode
    p = sqrt(p_squared)
    tmp2 = atan2((ca + cb), (d - sa - sb)) - atan2(2.0, p)
    t = mod2pi(alpha - tmp2)
    q = mod2pi(beta - tmp2)

    return t, p, q, mode


def RLR(alpha, beta, d):
    sa = sin(alpha)
    sb = sin(beta)
    ca = cos(alpha)
    cb = cos(beta)
    c_ab = cos(alpha - beta)

    mode = ['R', 'L', 'R']
    tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp_rlr) > 1.0:
        return None, None, None, mode

    p = mod2pi(2 * pi - acos(tmp_rlr))
    t = mod2pi(alpha - atan2(ca - cb, d - sa + sb) + mod2pi(p / 2.0))
    q = mod2pi(alpha - beta - t + mod2pi(p))
    return t, p, q, mode


def LRL(alpha, beta, d):
    sa = sin(alpha)
    sb = sin(beta)
    ca = cos(alpha)
    cb = cos(beta)
    c_ab = cos(alpha - beta)

    mode = ['L', 'R', 'L']
    tmp_lrl = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (- sa + sb)) / 8.0
    if abs(tmp_lrl) > 1:
        return None, None, None, mode
    p = mod2pi(2 * pi - acos(tmp_lrl))
    t = mod2pi(-alpha - atan2(ca - cb, d + sa - sb) + p / 2.0)
    q = mod2pi(mod2pi(beta) - alpha - t + mod2pi(p))

    return t, p, q, mode


def dubins_path(sx, sy, syaw, ex, ey, eyaw, c):

    ex = ex - sx
    ey = ey - sy

    dx = cos(syaw) * ex + sin(syaw) * ey
    dy = - sin(syaw) * ex + cos(syaw) * ey
    deyaw = eyaw - syaw

    D = sqrt(dx * dx + dy * dy)
    d = D * c

    theta = mod2pi(atan2(dy, dx))
    alpha = mod2pi(- theta)
    beta = mod2pi(deyaw - theta)

    planners = [LSL, RSR, LSR, RSL, RLR, LRL] # RLR and LRL could cause instability
    #planners = [LSL, RSR, LSR, RSL]
    bcost, bt, bp, bq, bmode = inf, inf, inf, inf, None

    for planner in planners:
        t, p, q, mode = planner(alpha, beta, d)
        if t is None:
            continue

        cost = (abs(t) + abs(p) + abs(q))
        if bcost > cost:
            bcost, bt, bp, bq, bmode = cost, t, p, q, mode

    return bcost, bt, bp, bq, bmode


def main():
    sx = 0
    sy = 0
    syaw = pi


    ex = 10
    ey = 0
    eyaw = 0

    v_x = 2.0
    v_theta = 1.0

    c = v_theta / v_x

    bcost, bt, bp, bq, bmode = dubins_path(sx, sy, syaw, ex, ey, eyaw, c)

    print(bt / v_theta)
    print(bp / c / v_x)
    print(bq / v_theta)
    print(bmode)
    print(bcost)



if __name__ == "__main__":
    main()