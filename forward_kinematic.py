import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from math import pi
import numpy as np
import matplotlib.pyplot as plt


# Function for visualization
def plot_arm(ax, T1, T2, T3, T4):
    def plot_axis(ax, x, y, z, x_a, y_a, z_a):
        ax.quiver(x, y, z, x_a[0], x_a[1], x_a[2], length=5, color='r', normalize=False)
        ax.quiver(x, y, z, y_a[0], y_a[1], y_a[2], length=5, color='g', normalize=False)
        ax.quiver(x, y, z, z_a[0], z_a[1], z_a[2], length=5, color='b', normalize=False)

    plot_axis(ax, 0, 0, 0, [1, 0, 0], [0, 1, 0], [0, 0, 1])
    T = T1
    plot_axis(ax, T[0, 3], T[1, 3], T[2, 3], T[0:3, 0], T[0:3, 1], T[0:3, 2])
    T = T * T2
    plot_axis(ax, T[0, 3], T[1, 3], T[2, 3], T[0:3, 0], T[0:3, 1], T[0:3, 2])
    T = T * T3
    plot_axis(ax, T[0, 3], T[1, 3], T[2, 3], T[0:3, 0], T[0:3, 1], T[0:3, 2])
    T = T * T4
    plot_axis(ax, T[0, 3], T[1, 3], T[2, 3], T[0:3, 0], T[0:3, 1], T[0:3, 2])


# Function to compute the forward kinematics
def fk(q):
    theta, theta1, theta2, theta3, theta4, theta5, theta6, theta7, a, d, d1, d3, d5, d7, alpha = \
        dynamicsymbols("theta theta1 theta2 theta3 theta4 theta5 theta6 theta7 a d d1 d3 d5 d7 alpha")

    T = sp.Matrix([[sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
                   [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.sin(alpha) * sp.cos(theta), a * sp.sin(theta)],
                   [0, sp.sin(alpha), sp.cos(alpha), d], [0, 0, 0, 1]])
# left
    t1 = T.subs({theta: 0, a: 2.5, d: 5.4 + q[0], alpha: -pi / 2})
    t2 = T.subs({theta: q[1], a: 0, d: 3.6, alpha: pi / 2})
    t3 = T.subs({theta: q[2], a: 6.9, d: 0, alpha: 0})
    t4 = T.subs({theta: q[3], a: 12.5, d: 0, alpha: 0})

# right
#    t1 = T.subs({theta: 0, a: 2.5, d: 5.4 + q[0], alpha: pi / 2})
#    t2 = T.subs({theta: q[1], a: 0, d: 3.6, alpha: -pi / 2})
#    t3 = T.subs({theta: q[2], a: 6.9, d: 0, alpha: 0})
#    t4 = T.subs({theta: q[3], a: 12.5, d: 0, alpha: 0})

    return t1, t2, t3, t4


fig = plt.figure()

# ---------plot the frame ---------------
q = np.array([2, 0, 0, 0])
q1 = np.array([3, 0, 0, np.pi / 4])
q2 = np.array([1, 0, np.pi / 4, np.pi / 4])
q3 = np.array([1, np.pi / 4, np.pi / 4, np.pi/4])
q4 = np.array([3, 0, 0, -np.pi / 4])
q5 = np.array([1, 0, np.pi / 4, -np.pi / 4])
q6 = np.array([1, np.pi / 4, np.pi / 4, np.pi / 4])

T1, T2, T3, T4 = fk(q)

ax = fig.add_subplot(projection='3d')

ax.set_xlim3d([-15, 15])
ax.set_xlabel('X')

ax.set_ylim3d([-15, 15])
ax.set_ylabel('Y')

ax.set_zlim3d([0, 30])
ax.set_zlabel('Z')

ax.view_init(elev=10.0, azim=60)

np_end_pos = T1 * T2 * T3 * T4
ax.scatter(np_end_pos[:, 0], np_end_pos[:, 1], np_end_pos[:, 2])

plot_arm(ax, T1, T2, T3, T4)
plt.show()
