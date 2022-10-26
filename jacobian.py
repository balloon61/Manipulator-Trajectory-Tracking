# This is a sample Python script.
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols, Point, ReferenceFrame
from math import pi
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse


# Function for visualization
def plot_arm(ax, T1, T2, T3, T4, T5, T6, T7):
    def plot_axis(ax, x, y, z, x_a, y_a, z_a):
        ax.quiver(x, y, z, x_a[0], x_a[1], x_a[2], length=0.1, color='r', normalize=False)
        ax.quiver(x, y, z, y_a[0], y_a[1], y_a[2], length=0.1, color='g', normalize=False)
        ax.quiver(x, y, z, z_a[0], z_a[1], z_a[2], length=0.1, color='b', normalize=False)

    plot_axis(ax, 0, 0, 0, [1, 0, 0], [0, 1, 0], [0, 0, 1])
    T = T1
    plot_axis(ax, T[0, 3], T[1, 3], T[2, 3], T[0:3, 0], T[0:3, 1], T[0:3, 2])
    T = T1 * T2
    plot_axis(ax, T[0, 3], T[1, 3], T[2, 3], T[0:3, 0], T[0:3, 1], T[0:3, 2])
    T = T1 * T2 * T3
    plot_axis(ax, T[0, 3], T[1, 3], T[2, 3], T[0:3, 0], T[0:3, 1], T[0:3, 2])
    T = T1 * T2 * T3 * T4
    plot_axis(ax, T[0, 3], T[1, 3], T[2, 3], T[0:3, 0], T[0:3, 1], T[0:3, 2])
    T = T1 * T2 * T3 * T4 * T5
    plot_axis(ax, T[0, 3], T[1, 3], T[2, 3], T[0:3, 0], T[0:3, 1], T[0:3, 2])
    T = T1 * T2 * T3 * T4 * T5 * T6
    plot_axis(ax, T[0, 3], T[1, 3], T[2, 3], T[0:3, 0], T[0:3, 1], T[0:3, 2])
    T = T1 * T2 * T3 * T4 * T5 * T6 * T7
    plot_axis(ax, T[0, 3], T[1, 3], T[2, 3], T[0:3, 0], T[0:3, 1], T[0:3, 2])


# Function to compute the forward kinematics
def forward(q):
    theta, theta1, theta2, theta3, theta4, theta5, theta6, theta7, a, d, d1, d3, d5, d7, alpha = \
        dynamicsymbols("theta theta1 theta2 theta3 theta4 theta5 theta6 theta7 a d d1 d3 d5 d7 alpha")

    T = sp.Matrix([[sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
                   [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.sin(alpha) * sp.cos(theta), a * sp.sin(theta)],
                   [0, sp.sin(alpha), sp.cos(alpha), d], [0, 0, 0, 1]])
    T1 = T.subs({theta: q[0], a: 0, d: 0.4, alpha: pi / 2})
    T2 = T.subs({theta: q[1], a: 0, d: 0, alpha: -pi / 2})
    T3 = T.subs({theta: 0, a: 0, d: 0.38, alpha: -pi / 2})
    T4 = T.subs({theta: q[2], a: 0, d: 0, alpha: pi / 2})
    T5 = T.subs({theta: q[3], a: 0, d: 0.3995, alpha: pi / 2})
    T6 = T.subs({theta: q[4], a: 0, d: 0, alpha: -pi / 2})
    T7 = T.subs({theta: q[5], a: 0, d: 0.2055, alpha: 0})

    return T1, T2, T3, T4, T5, T6, T7


def Jacobian(T1, T2, T3, T4, T5, T6, T7):
    T01 = T1
    T02 = T01 * T2
    T03 = T02 * T3
    T04 = T03 * T4
    T05 = T04 * T5
    T06 = T05 * T6
    T07 = T06 * T7
    z0 = np.array([0, 0, 1])
    o0 = np.array([0, 0, 0])

    z1 = np.array(T01[0:3, 2]).astype(np.float64).squeeze(-1)
    o1 = np.array(T01[0:3, 3]).astype(np.float64).squeeze(-1)

    z2 = np.array(T02[0:3, 2]).astype(np.float64).squeeze(-1)
    o2 = np.array(T02[0:3, 3]).astype(np.float64).squeeze(-1)

    z3 = np.array(T03[0:3, 2]).astype(np.float64).squeeze(-1)
    o3 = np.array(T03[0:3, 3]).astype(np.float64).squeeze(-1)

    z4 = np.array(T04[0:3, 2]).astype(np.float64).squeeze(-1)
    o4 = np.array(T04[0:3, 3]).astype(np.float64).squeeze(-1)

    z5 = np.array(T05[0:3, 2]).astype(np.float64).squeeze(-1)
    o5 = np.array(T05[0:3, 3]).astype(np.float64).squeeze(-1)

    z6 = np.array(T06[0:3, 2]).astype(np.float64).squeeze(-1)
    o6 = np.array(T06[0:3, 3]).astype(np.float64).squeeze(-1)

    o7 = np.array(T07[0:3, 3]).astype(np.float64).squeeze(-1)

    J1 = np.concatenate([np.cross(z0, o7 - o0), z0], 0)
    J2 = np.concatenate([np.cross(z1, o7 - o1), z1], 0)
    J3 = np.concatenate([np.cross(z2, o7 - o2), z2], 0)
    J4 = np.concatenate([np.cross(z3, o7 - o3), z3], 0)
    J5 = np.concatenate([np.cross(z4, o7 - o4), z4], 0)
    J6 = np.concatenate([np.cross(z5, o7 - o5), z5], 0)
    J7 = np.concatenate([np.cross(z6, o7 - o6), z6], 0)
    J = np.array([J1, J2, J4, J5, J6, J7]).T
    print(J)
    return J


# Time to draw the circle
max_time = 5

# Initial orientation (for the cylindrical coordinate)
theta_w = pi / 2

# Initial configuration for each joint [q1, q2, q4, q5, q6, q7]
q = np.array([pi / 2, 0, pi / 2, pi / 2, 0, 0])

# Angular velocity to plot the circle(for the cylindrical coordiante)
omega = pi * 2 / max_time

fig = plt.figure()

# Time step
dt = 0.05

# Save the end effector positions (for visualization)
end_positions = []

radius = 0.1

for t in np.arange(0, max_time, dt):
    # Since each plot is cleared for each step, the axis is redefined
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim3d([-0.4, 0.4])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, 0.8])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0, 0.8])
    ax.set_zlabel('Z')

    ax.view_init(elev=10.0, azim=60)
    # ----------------------------------------------------

    # Forward kinematics given the joint angles
    T1, T2, T3, T4, T5, T6, T7 = forward(q)
    # Calculate the Jacobian for inverse kinematics
    J = Jacobian(T1, T2, T3, T4, T5, T6, T7)
    T = T1 * T2 * T3 * T4 * T5 * T6 * T7

    # The desired derivative in the current position
    dedt = np.array([-radius * np.sin(theta_w) * omega, 0, radius * np.cos(theta_w) * omega, 0, 0, 0])

    # try:
    #    J_ = np.linalg.inv(J)
    #    dqdt = J_@dedt

    # dqdt = J_@dp_dt

    # Obtain the angular velocity for each joint by solving J*dqdt = dp_dt
    # Using least square gives better results
    dqdt = np.linalg.lstsq(J, dedt)[0]

    # Update joint angles with q = q + dqdt*dt
    q += dqdt * dt
    theta_w += omega * dt

    # ----------- Visualization ------
    plot_arm(ax, T1, T2, T3, T4, T5, T6, T7)
    end_positions.append(np.array([float(T[0, 3]), float(T[1, 3]), float(T[2, 3])]))
    np_end_pos = np.array(end_positions)
    # for end_pos in end_positions:
    ax.scatter(np_end_pos[:, 0], np_end_pos[:, 1], np_end_pos[:, 2])
    plt.pause(dt)
    plt.clf()

# ---------plot the final frame ---------------
ax = fig.add_subplot(projection='3d')

ax.set_xlim3d([-0.4, 0.4])
ax.set_xlabel('X')

ax.set_ylim3d([0, 0.8])
ax.set_ylabel('Y')

ax.set_zlim3d([0, 0.8])
ax.set_zlabel('Z')

ax.view_init(elev=10.0, azim=60)

np_end_pos = np.array(end_positions)
ax.scatter(np_end_pos[:, 0], np_end_pos[:, 1], np_end_pos[:, 2])

plot_arm(ax, T1, T2, T3, T4, T5, T6, T7)
plt.show()
