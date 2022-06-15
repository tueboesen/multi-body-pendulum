import math
import torch
torch.set_default_dtype(torch.float64)
import time

class NPendulum:
    def __init__(self,n,dt):
        """
        Initiates an n-pendulum class for simulating pendulums
        :param n:
        :param dt:
        """
        self.n = n
        self.A = torch.zeros((n,n))
        self.b = torch.zeros((n))
        self.x = torch.zeros((n))
        self.g = 9.82
        self.dt = dt

        C = torch.zeros(n,n)
        for i in range(n):
            for j in range(n):
                C[i,j] = self.c(i,j)
        self.C = C

    def c(self,i,j):
        return self.n - max(i,j)

    def npendulum(self,theta,dtheta):
        """
        Sets up and solves the n-pendulum linear system of equations
        :param theta:
        :param dtheta:
        :return:
        """
        n = self.n
        A = self.A
        b = self.b
        g = self.g

        for i in range(n):
            for j in range(n):
                A[i, j] = self.c(i, j) * torch.cos(theta[i]-theta[j])

        for i in range(n):
            tmp = 0
            for j in range(n):
                tmp = tmp - self.c(i,j) * dtheta[j]*dtheta[j] * torch.sin(theta[i] - theta[j])
            b[i] = tmp - g * (n - i + 1) *torch.sin(theta[i])

        ddtheta = torch.linalg.solve(A,b)
        return dtheta,ddtheta

    def npendulum_fast(self,theta,dtheta):
        """
        Faster version of npendulum
        :param theta:
        :param dtheta:
        :return:
        """
        n = self.n
        g = self.g

        A = self.C * torch.cos(theta[:,None] - theta[None,:])

        B = - self.C * (dtheta[None,:]*dtheta[None,:]) * torch.sin(theta[:,None] - theta[None,:])
        b = torch.sum(B,dim=1)
        b = b - g * torch.arange(n,0,-1) * torch.sin(theta)
        ddtheta = torch.linalg.solve(A,b)
        return dtheta,ddtheta



    def rk4(self,dt,theta,dtheta):
        """
        Runge kutta 4 integration
        :param dt:
        :param theta:
        :param dtheta:
        :return:
        """
        k1 = self.npendulum_fast(theta,dtheta)
        k2 = self.npendulum_fast(theta+dt/2*k1[0],dtheta+dt/2*k1[1])
        k3 = self.npendulum_fast(theta+dt/2*k2[0],dtheta+dt/2*k2[1])
        k4 = self.npendulum_fast(theta+dt*k3[0],dtheta+dt*k3[1])


        theta  = theta + dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        dtheta = dtheta + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

        return theta,dtheta

    def simulate(self,nsteps,theta,dtheta):
        """
        Simulates an n-pendulum.
        :param nsteps:
        :param theta:
        :param dtheta:
        :return:
        """
        dt = self.dt
        n = self.n
        thetas = torch.zeros(n,nsteps+1)
        dthetas = torch.zeros(n,nsteps+1)
        thetas[:, 0] = theta
        dthetas[:, 0] = dtheta
        t = torch.linspace(0,nsteps * dt, nsteps + 1)

        for i in range(nsteps):
            theta, dtheta = self.rk4(dt,theta,dtheta)
            thetas[:,i+1] = theta
            dthetas[:,i+1] = dtheta
        return t, thetas, dthetas

    def get_coordinates_from_angle(self,theta):
        """
        Converts angles to cartesian coordinates
        :param theta:
        :return:
        """
        n,ns = theta.shape
        x = torch.zeros(n+1,ns)
        y = torch.zeros(n+1,ns)
        for i in range(n):
            x[i+1] = x[i] + torch.sin(theta[i])
            y[i+1] = y[i] - torch.cos(theta[i])
        return x, y

def animate_pendulum(x,y,save=None):
    """
    Animates the pendulum.
    If save is None it will show a movie in python
    otherwise it will save a gif to the location specified in save
    """

    import matplotlib;
    matplotlib.use("TkAgg")
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    n,nsteps = x.shape

    line, = ax.plot(x[:,0], y[:,0])
    line2, = ax.plot(x[:,0], y[:,0], 'ro')
    lines = [line,line2]

    def init():
        ax.set_xlim(-n,n)
        ax.set_ylim(-n,n)
        return lines

    def update(i):
        for ln in lines:
            ln.set_xdata(x[:,i])  # update the data
            ln.set_ydata(y[:,i])
        return lines

    ani = animation.FuncAnimation(fig, update, frames=np.arange(1, nsteps), init_func=init, interval=25, blit=True)
    if save is None:
        plt.show()
    else:
        writergif = animation.PillowWriter(fps=30)
        ani.save(save, writer=writergif)


if __name__ == '__main__':
    n = 5
    dt = 0.01
    Npend = NPendulum(n,dt)

    theta0 = 0.5*math.pi*torch.ones(n)
    dtheta0 = 0.0*torch.ones(n)
    nsteps = 1000

    t0 = time.time()
    times, thetas, dthetas = Npend.simulate(nsteps,theta0,dtheta0)
    t1 = time.time()
    print(f"simulated {nsteps} steps for a {n}-pendulum in {t1-t0:2.2f}s")

    x,y = Npend.get_coordinates_from_angle(thetas)

    animate_pendulum(x.numpy(), y.numpy())

    # file = f'/home/tue/{n}_pendulum_{nsteps}.gif'
    # animate_pendulum(x.numpy(), y.numpy(),save=file)

