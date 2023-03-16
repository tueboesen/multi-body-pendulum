import math
from math import pi

import pytest
import torch

from src.n_pendulum import MultiBodyPendulum

def test_stability():
    """
    We should test that the accumulated error of the RK4 method scales as o(n**4)
    But for now we just test that the energy drift decreases as the step size goes down
    """
    n = 10
    dts = [0.02,0.01,0.005,0.0025,0.00125]
    theta0 = 0.5 * math.pi * torch.ones(n)
    dtheta0 = 0.0 * torch.ones(n)
    nsteps = 1000
    Edrift = []
    g = 9.82
    for dt in dts:
        mbp = MultiBodyPendulum(n, dt,g=g)
        nsteps = round(2/dt)
        mbp.simulate(nsteps=nsteps,theta_start=theta0,dtheta_start=dtheta0)
        E = mbp.energy
        En = (E - E[0]).abs()
        Emean = En.mean()
        Edrift.append(Emean.item())
    Edrift = torch.tensor(Edrift)
    assert (torch.diff(Edrift) < 0).all()


class TestAngleCartesian:

    n = 3
    dt = 0.001
    g = 9.82
    mbp = MultiBodyPendulum(n, dt,g=g)
    @pytest.mark.parametrize("x, y, vx, vy, theta, dtheta, error", [
        (0.0, -1.0, 0.0, 0.0, 0.0, 0.0, None),
        (1.0, 0.0, 0.0, 0.0, pi/2.0, 0.0, None),
        (1.0, 0.0, 0.0, 5.0, pi/2.0, 5.0, None),
        (1, 0.0, 0.0, 0.0, pi/2.0, 0.0, True),
    ])


    def test_known_angles_and_coordinates(self, x,y,vx,vy,theta,dtheta,error):
        x = torch.tensor([x])
        y = torch.tensor([y])
        vx = torch.tensor([vx])
        vy = torch.tensor([vy])
        theta = torch.tensor([theta])
        dtheta = torch.tensor([dtheta])

        if error is not None:
            with pytest.raises(Exception) as e_info:
                x1,y1,vx1,vy1 = self.mbp.get_coordinates_from_angles(theta,dtheta)
                theta1,dtheta1 = self.mbp.get_angles_from_coordinates(x,y,vx,vy)
        else:
            x1, y1, vx1, vy1 = self.mbp.get_coordinates_from_angles(theta, dtheta)
            theta1, dtheta1 = self.mbp.get_angles_from_coordinates(x, y, vx, vy)
            assert x == pytest.approx(x1)
            assert y == pytest.approx(y1)
            assert vx == pytest.approx(vx1)
            assert vy == pytest.approx(vy1)
            assert theta == pytest.approx(theta1)
            assert dtheta == pytest.approx(dtheta1)

    def test_angles_and_cartesian_are_reversible(self):
        # npenduls = torch.randint(1,100,10)
        # for n in npenduls:
        theta = (2*torch.rand(10,100))*pi
        dtheta = (2*torch.rand(10,100))*10
        (x, y, vx, vy) = self.mbp.get_coordinates_from_angles(theta, dtheta)
        t1, dt1 = self.mbp.get_angles_from_coordinates(x, y, vx, vy)
        assert theta == pytest.approx(t1)
        assert dtheta == pytest.approx(dtheta)

