import sympy as sp
from sympy.physics.mechanics import dynamicsymbols, Point, ReferenceFrame
from math import pi
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def plot_arm(ax, T_list):
    def plot_axis(ax, x, y, z, x_a, y_a, z_a):
        ax.quiver(x, y, z, x_a[0], x_a[1], x_a[2], length=1, color='r', normalize=False)
        ax.quiver(x, y, z, y_a[0], y_a[1], y_a[2], length=1, color='g', normalize=False)
        ax.quiver(x, y, z, z_a[0], z_a[1], z_a[2], length=1, color='b', normalize=False)

    plot_axis(ax, 0, 0, 0, [1, 0, 0], [0, 1, 0], [0, 0, 1])
    T = sp.eye(4)
    for T_tmp in T_list:
        T = T * T_tmp
        plot_axis(ax, T[0, 3], T[1, 3], T[2, 3], T[0:3, 0], T[0:3, 1], T[0:3, 2])


def forward(q):
    theta, a, d, alpha = dynamicsymbols("theta a d alpha")

    T = sp.Matrix([[sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
                   [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.sin(alpha) * sp.cos(theta), a * sp.sin(theta)],
                   [0, sp.sin(alpha), sp.cos(alpha), d], [0, 0, 0, 1]])

    T1 = T.subs({theta: 0, a: 2.5, d: 5.4 + q[0], alpha: -pi / 2})
    T2 = T.subs({theta: q[1], a: 0, d: 3.6, alpha: pi / 2})
    T3 = T.subs({theta: q[2], a: 6.9, d: 2, alpha: 0})
    T4 = T.subs({theta: q[3], a: 12.5, d: 0, alpha: 0})
    return [T1, T2, T3, T4]


def forward1(q):
    theta, a, d, alpha = dynamicsymbols("theta a d alpha")

    T = sp.Matrix([[sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
                   [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.sin(alpha) * sp.cos(theta), a * sp.sin(theta)],
                   [0, sp.sin(alpha), sp.cos(alpha), d], [0, 0, 0, 1]])

    T1 = T.subs({theta: 0, a: 2.5, d: 5.4 + q[0], alpha: pi / 2})
    T2 = T.subs({theta: q[1], a: 0, d: 3.6, alpha: pi / 2})
    T3 = T.subs({theta: q[2], a: 6.9, d: 2, alpha: 0})
    T4 = T.subs({theta: q[3], a: 12.5, d: 0, alpha: 0})
    return [T1, T2, T3, T4]


def Jacobian(T1, T2, T3, T4):
    T01 = T1
    T02 = T01 * T2
    T03 = T02 * T3
    T04 = T03 * T4

    z0 = np.array([0, 0, 1])
    o0 = np.array([0, 0, 0])

    z1 = np.array(T01[0:3, 2]).astype(np.float64).squeeze(-1)
    o1 = np.array(T01[0:3, 3]).astype(np.float64).squeeze(-1)

    z2 = np.array(T02[0:3, 2]).astype(np.float64).squeeze(-1)
    o2 = np.array(T02[0:3, 3]).astype(np.float64).squeeze(-1)

    z3 = np.array(T03[0:3, 2]).astype(np.float64).squeeze(-1)
    o3 = np.array(T03[0:3, 3]).astype(np.float64).squeeze(-1)

    o4 = np.array(T04[0:3, 3]).astype(np.float64).squeeze(-1)

    J1 = np.zeros(6)
    # J1[0:3] = z0
    J2 = np.concatenate([np.cross(z1, o4 - o1), z1], 0)
    J3 = np.concatenate([np.cross(z2, o4 - o2), z2], 0)
    J4 = np.concatenate([np.cross(z3, o4 - o3), z3], 0)

    J = np.array([J1, J2, J3, J4]).T
    return J


q = np.array([0.0, pi / 8, pi / 8 * 3, -pi / 2])
T_list = forward(q)
d = 0
end_positions = []

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

dt = 0.1
for i in np.arange(0, 5, dt):
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d([0, 20.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, 20.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0, 20.0])
    ax.set_zlabel('Z')

    T_list = forward(q)
    J = Jacobian(T_list[0], T_list[1], T_list[2], T_list[3])
    T = T_list[0] * T_list[1] * T_list[2] * T_list[3]

    dedt = np.array([0, 0, 1, 0, 0, 0]).astype(np.float)
    dqdt = np.linalg.lstsq(J, dedt)[0]
    print(q)

    q += dqdt * dt
    plot_arm(ax, T_list)

    end_positions.append(np.array([float(T[0, 3]), float(T[1, 3]), float(T[2, 3])]))
    np_end_pos = np.array(end_positions)
    ax.scatter(np_end_pos[:, 0], np_end_pos[:, 1], np_end_pos[:, 2])
    plt.pause(0.01)
    plt.clf()
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d([0, 20.0])
ax.set_xlabel('X')

ax.set_ylim3d([0, 20.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0, 20.0])
ax.set_zlabel('Z')
np_end_pos = np.array(end_positions)
plot_arm(ax, T_list)
ax.scatter(np_end_pos[:, 0], np_end_pos[:, 1], np_end_pos[:, 2])
plt.show()