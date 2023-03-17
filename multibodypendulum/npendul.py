import math
import os
from random import randint

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
import time


class MultiBodyPendulum:
    """
    A multibody pendulum class, which enables the simulation of a pendulum with massless rigid strings between n point masses.

    Note that this code only supports pendulums where all pendulums have a length of 1m and mass of 1kg.
    """

    def __init__(self, npendulums: int, dt: float, g: float = 9.82):
        """
        Initiates an n-pendulum class for simulating multi-body pendulums.
        See https://travisdoesmath.github.io/pendulum-explainer/ or https://github.com/tueboesen/n-pendulum for the equations,
        :param npendulums: number of pendulums
        :param dt: time stepping size
        """
        self.n = npendulums
        self.g = g
        self.dt = dt

        self.thetas = None
        self.dthetas = None

        self._A = torch.zeros((npendulums, npendulums))  # Memory allocations
        self._b = torch.zeros((npendulums))
        self._x = torch.zeros((npendulums))

        C = torch.zeros(
            npendulums, npendulums
        )  # A lookup matrix that we use for fast calculation of static information rather than actually compute this in every step
        for i in range(npendulums):
            for j in range(npendulums):
                C[i, j] = self._c(i, j)
        self._C = C

    @property
    def xy(self):
        """
        Converts the angle to cartisian coordinates
        """
        thetas = self.thetas
        n = thetas.shape[0]
        x = torch.empty_like(thetas)
        y = torch.empty_like(thetas)
        x[0] = torch.sin(thetas[0])
        y[0] = -torch.cos(thetas[0])
        for i in range(1, n):
            x[i] = x[i - 1] + torch.sin(thetas[i])
            y[i] = y[i - 1] - torch.cos(thetas[i])
        return x, y

    @property
    def vxy(self):
        """
        Converts the angle and angle velocity to cartesian velocity
        """
        thetas = self.thetas
        dthetas = self.dthetas
        n = dthetas.shape[0]
        vx = torch.empty_like(dthetas)
        vy = torch.empty_like(dthetas)
        vx[0] = dthetas[0] * torch.cos(thetas[0])
        vy[0] = dthetas[0] * torch.sin(thetas[0])
        for i in range(1, n):
            vx[i] = vx[i - 1] + dthetas[i] * torch.cos(thetas[i])
            vy[i] = vy[i - 1] + dthetas[i] * torch.sin(thetas[i])
        return vx, vy

    @property
    def energy_kinetic(self):
        vx, vy = self.vxy
        K = 0.5 * torch.sum(vx**2 + vy**2, dim=0)
        return K

    @property
    def energy_potential(self):
        x, y = self.xy
        V = self.g * torch.sum(y, dim=0)
        return V

    @property
    def energy(self):
        E = self.energy_potential + self.energy_kinetic
        return E

    def energy_drift(self):
        """
        Calculates the average drift in energy over the entire simulation.
        """
        E = self.energy
        En = (E - E[0]).abs()
        Emean = En.mean()
        return Emean

    @staticmethod
    def get_angles_from_coordinates(x, y, vx, vy):
        """
        Converts cartesian coordinates and velocities to angles.
        Expects the input to have shape [npendulums,...]
        """
        assert torch.is_floating_point(
            x
        )  # We explicitly test for this since the code will run for integers, but produce nonsensible results
        assert torch.is_floating_point(
            y
        )  # We explicitly test for this since the code will run for integers, but produce nonsensible results
        assert torch.is_floating_point(
            vx
        )  # We explicitly test for this since the code will run for integers, but produce nonsensible results
        assert torch.is_floating_point(
            vy
        )  # We explicitly test for this since the code will run for integers, but produce nonsensible results

        assert x.shape == y.shape == vx.shape == vy.shape
        thetas = torch.empty_like(x)
        dthetas = torch.empty_like(x)
        thetas[0] = torch.atan2(y[0], x[0])
        for i in range(1, x.shape[0]):
            thetas[i] = torch.atan2(y[i] - y[i - 1], x[i] - x[i - 1])
        thetas += math.pi / 2
        thetas = torch.remainder(thetas, 2 * math.pi)  # - math.pi
        M = torch.isclose(torch.cos(thetas[0]), torch.zeros_like(thetas[0]))
        if M.all() == True:
            dthetas[0] = vy[0] / torch.sin(thetas[0])
        else:
            dthetas[0, M] = vx[0, M] / torch.cos(thetas[0, M])
            M = ~M
            dthetas[0, M] = vx[0, M] / torch.cos(thetas[0, M])
        for i in range(1, x.shape[0]):
            M = torch.isclose(torch.cos(thetas[i]), torch.zeros_like(thetas[i]))
            if M.all() == True:
                dthetas[i] = (vy[i] - vy[i - 1]) / torch.sin(thetas[i])
            else:
                dthetas[i, M] = (vy[i, M] - vy[i - 1, M]) / torch.sin(thetas[i, M])
                M = ~M
                dthetas[i, M] = (vx[i, M] - vx[i - 1, M]) / torch.cos(thetas[i, M])
        return thetas, dthetas

    @staticmethod
    def get_coordinates_from_angles(thetas, dthetas):
        """
        Converts angles and angle velocities into cartesian coordinates
        """
        assert torch.is_floating_point(
            thetas
        )  # We explicitly test for this since the code will run for integers, but produce nonsensible results
        assert torch.is_floating_point(
            dthetas
        )  # We explicitly test for this since the code will run for integers, but produce nonsensible results
        n = dthetas.shape[0]
        x = torch.empty_like(thetas)
        y = torch.empty_like(thetas)
        vx = torch.empty_like(dthetas)
        vy = torch.empty_like(dthetas)

        x[0] = torch.sin(thetas[0])
        y[0] = -torch.cos(thetas[0])
        vx[0] = dthetas[0] * torch.cos(thetas[0])
        vy[0] = dthetas[0] * torch.sin(thetas[0])
        for i in range(1, n):
            x[i] = x[i - 1] + torch.sin(thetas[i])
            y[i] = y[i - 1] - torch.cos(thetas[i])
            vx[i] = vx[i - 1] + dthetas[i] * torch.cos(thetas[i])
            vy[i] = vy[i - 1] + dthetas[i] * torch.sin(thetas[i])
        return x, y, vx, vy

    def _c(self, i: int, j: int):
        return self.n - max(i, j)

    def extend_tensor(self, x):
        """
        A simple helper function that prepends the first dimension of a tensor by 1 with value 0 (This is useful when you want to have a multi-body-pendulum including origo)
        """
        shape = list(x.shape)
        shape[0] += 1
        x2 = torch.zeros(shape, device=x.device, dtype=x.dtype)
        x2[1:] = x
        return x2

    def step_slow(self, theta, dtheta):
        """
        Performs a single step for a multibody pendulum.

        More specifically it sets up and solves the n-pendulum linear system of equations in angular coordinates where the constraints are naturally obeyed.

        Note this function should not be used. Use npendulum_fast instead since it is an optimized version of this. This function is still here for comparison and to easily understand the calculations.

        :param theta: angular coordinate
        :param dtheta: angular velocity
        :return:
        """
        n = self.n
        A = self._A
        b = self._b
        g = self.g

        for i in range(n):
            for j in range(n):
                A[i, j] = self._c(i, j) * torch.cos(theta[i] - theta[j])

        for i in range(n):
            tmp = 0
            for j in range(n):
                tmp = tmp - self._c(i, j) * dtheta[j] * dtheta[j] * torch.sin(
                    theta[i] - theta[j]
                )
            b[i] = tmp - g * (n - i + 1) * torch.sin(theta[i])

        ddtheta = torch.linalg.solve(A, b)
        return dtheta, ddtheta

    def step(self, theta, dtheta):
        """
        Faster version of step_slow
        """
        n = self.n
        g = self.g

        A = self._C * torch.cos(theta[:, None] - theta[None, :])

        B = (
            -self._C
            * (dtheta[None, :] * dtheta[None, :])
            * torch.sin(theta[:, None] - theta[None, :])
        )
        b = torch.sum(B, dim=1)
        b = b - g * torch.arange(n, 0, -1) * torch.sin(theta)
        ddtheta = torch.linalg.solve(A, b)
        return dtheta, ddtheta

    def rk4(self, dt, theta, dtheta):
        """
        Runge kutta 4 integration
        :param dt:
        :param theta:
        :param dtheta:
        :return:
        """
        k1 = self.step(theta, dtheta)
        k2 = self.step(theta + dt / 2 * k1[0], dtheta + dt / 2 * k1[1])
        k3 = self.step(theta + dt / 2 * k2[0], dtheta + dt / 2 * k2[1])
        k4 = self.step(theta + dt * k3[0], dtheta + dt * k3[1])

        theta = theta + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        dtheta = dtheta + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

        return theta, dtheta

    def simulate(
        self,
        nsteps: int,
        theta_start: torch.FloatTensor,
        dtheta_start: torch.FloatTensor,
    ):
        """
        Simulates an n-pendulum.
        :param nsteps:
        :param theta:
        :param dtheta:
        :return:
        """
        dt = self.dt
        n = self.n
        self.thetas = torch.zeros(n, nsteps + 1)
        self.dthetas = torch.zeros(n, nsteps + 1)
        self.thetas[:, 0] = theta_start
        self.dthetas[:, 0] = dtheta_start
        t = torch.linspace(0, nsteps * dt, nsteps + 1)
        theta = theta_start
        dtheta = dtheta_start

        for i in range(nsteps):
            theta, dtheta = self.rk4(dt, theta, dtheta)
            self.thetas[:, i + 1] = theta
            self.dthetas[:, i + 1] = dtheta
        return t, self.thetas, self.dthetas

    def animate_pendulum(self, delay_between_frames=2, file=None):
        """
        Animates the pendulum.
        If save is None it will show a movie in python
        otherwise it will save a gif to the location specified in save
        """
        x0, y0 = self.xy
        vx0, vy0 = self.vxy
        x = self.extend_tensor(x0).numpy()
        y = self.extend_tensor(y0).numpy()
        vx = self.extend_tensor(vx0).numpy()
        vy = self.extend_tensor(vy0).numpy()

        fig, ax = plt.subplots()
        n, nsteps = x.shape

        (line,) = ax.plot(x[:, 0], y[:, 0])
        (line2,) = ax.plot(x[:, 0], y[:, 0], "ro")
        lines = [line, line2]

        v_origo = np.asarray([x[1:, 0], y[1:, 0]])
        arrows = plt.quiver(
            *v_origo, vx[1:, 0], vy[1:, 0], color="g", scale=100, width=0.003
        )
        lines.append(arrows)

        def init():
            ax.set_xlim(-n, n)
            ax.set_ylim(-n, n)
            return lines

        def update(i):
            for j in range(2):
                lines[j].set_xdata(x[:, i])
                lines[j].set_ydata(y[:, i])
            origo = np.asarray([x[1:, i], y[1:, i]]).transpose()
            lines[-1].set_offsets(origo)
            lines[-1].set_UVC(vx[1:, i], vy[1:, i])
            return lines

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=np.arange(1, nsteps),
            init_func=init,
            interval=delay_between_frames,
            blit=True,
        )
        if file is None:
            plt.show()
        else:
            writergif = animation.PillowWriter(fps=30)
            ani.save(file, writer=writergif)
        return (fig, ax)

    @staticmethod
    def plot_pendulum(
        x,
        y,
        vx=None,
        vy=None,
        fighandler=None,
        color_velocity="red",
        color_pendulums="blue",
        color_strings="grey",
        file=None,
    ):
        """
        Plots a snapshot figure of the pendulum. Note that this assumes that origo is prepended to the data
        """
        if vx is None or vy is None:
            assert (
                vx == vy
            ), "If vx or vy is not None, then both of them have to be given."
        else:
            vx = vx.numpy()
            vy = vy.numpy()
        if fighandler is None:
            fig, ax = plt.subplots(figsize=(15, 15))
        else:
            fig, ax = fighandler
        x = x.numpy()
        y = y.numpy()

        # v_origo = np.asarray([Rin[1:, 0], Rin[1:, 1]])
        if vx is not None:
            plt.quiver(x, y, vx, vy, color=color_velocity, scale=100, width=0.003)

        (l_in,) = ax.plot(x, y, color=color_strings, alpha=0.2)
        (lm_in,) = ax.plot(x, y, "o", color=color_pendulums, alpha=0.7, ms=20)

        ax.set_xlim([-x.shape[0], x.shape[0]])
        ax.set_ylim([-x.shape[0], x.shape[0]])
        # plt.axis('square')
        plt.axis("off")
        if file is not None:
            os.makedirs(os.path.dirname(file), exist_ok=True)
            plt.savefig(file, bbox_inches="tight", pad_inches=0)
            plt.close()
        return (fig, ax)

    def plot_energy(self, file=None):
        """
        Plots the kinetic, potential and total energy. Also plots the total energy separately so energy drift over the simulation can easily be seen.
        """
        import matplotlib.pyplot as plt

        plt.subplot(211)
        K = self.energy_kinetic
        V = self.energy_potential
        E = self.energy
        plt.plot(E, label="Energy")
        plt.plot(K, label="Kinetic energy")
        plt.plot(V, label="Potential energy")
        plt.legend()
        plt.subplot(212)
        plt.plot(E, label="Energy")
        plt.legend()

        if file is not None:
            os.makedirs(os.path.dirname(file), exist_ok=True)
            plt.savefig(file)
            plt.close()
        else:
            plt.show()
            plt.pause(0.1)
        return


if __name__ == "__main__":
    n = 5
    dt = 0.01
    g = 9.82
    mbp = MultiBodyPendulum(n, dt, g=g)
    theta0 = 0.5 * math.pi * torch.ones(n)
    dtheta0 = 0.0 * torch.ones(n)
    nsteps = 1000

    t0 = time.time()
    times, thetas, dthetas = mbp.simulate(nsteps, theta0, dtheta0)
    t1 = time.time()
    print(f"simulated {nsteps} steps for a {n}-pendulum in {t1-t0:2.2f}s")
    mbp.plot_energy()  # file="../docs/energy.png"
    mbp.animate_pendulum()
    x, y = mbp.xy
    vx, vy = mbp.vxy
    x2 = mbp.extend_tensor(x)
    y2 = mbp.extend_tensor(y)
    vx2 = mbp.extend_tensor(vx)
    vy2 = mbp.extend_tensor(vy)

    idx = randint(0, nsteps - 1)
    mbp.plot_pendulum(
        x2[:, idx], y2[:, idx], vx2[:, idx], vy2[:, idx]
    )  # file='./pendulum-snapshot.png'
