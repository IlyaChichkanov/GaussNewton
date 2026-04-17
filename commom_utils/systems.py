import casadi as ca
from casadi import SX, vertcat
import numpy as np
from abc import ABC, abstractmethod

from jax import jit
from jax import numpy as jnp
from commom_utils.ode_system import ODESystem

def get_interpolation_symbolic(x_grid, x, name='y_values'):
    n_points = len(x_grid)
    y_values = ca.SX.sym(name, n_points)

    def get_basis_functions(x, x_grid):
        basis = []
        for i in range(n_points):
            # Create triangular basis function
            if i == 0:
                b = ca.fmax(0, (x_grid[1] - x)/(x_grid[1] - x_grid[0]))
            elif i == n_points-1:
                b = ca.fmax(0, (x - x_grid[-2])/(x_grid[-1] - x_grid[-2]))
            else:
                left = (x - x_grid[i-1])/(x_grid[i] - x_grid[i-1])
                right = (x_grid[i+1] - x)/(x_grid[i+1] - x_grid[i])
                b = ca.fmax(0, ca.fmin(left, right))
            basis.append(b)
        return basis


    basis = get_basis_functions(x, x_grid)
    interp_value = 0
    for i in range(n_points):
        interp_value += y_values[i] * basis[i]

    f_interp = ca.Function('param_interp', 
                        [x, y_values],
                        [interp_value])
    return f_interp, y_values




class Pendulum(ODESystem):
    def __init__(self, order=1):
        super().__init__(nx=4, np=3, nu=1)

    def get_derivative(self, state, theta, u):
        g = 10
        M, m, l = theta[0], theta[1], theta[2]
        x1, theta, v1, dtheta = state[0], state[1], state[2], state[3]
        F = u[0]
        denominator = M + m - m*np.cos(theta)*np.cos(theta)
        f_expl = vertcat(v1,
                        dtheta,
                        (-m*l*np.sin(theta)*dtheta*dtheta + m*g*np.cos(theta)*np.sin(theta)+F)/denominator,
                        (-m*l*np.cos(theta)*np.sin(theta)*dtheta*dtheta + F*np.cos(theta)+(M+m)*g*np.sin(theta))/(l*denominator))

        return f_expl


    def observation(self, state, theta, u):
        x1, theta, v1, dtheta = state[0], state[1], state[2], state[3]
        return x1, theta
        
    def get_input_signals(self, t):
        return [jnp.sin(0.2*t)]


class DelaySystem(ODESystem):
    def __init__(self, order=1):
        self.order = order
        if order == 1:
            super().__init__(nx=1, np=1, nu=1)
        else:
            super().__init__(nx=2, np=1, nu=1)

    def get_derivative(self, state, theta, u):
        tau = theta[0]
        tau_safe = ca.fmax(tau, 1e-6)
        if self.order == 1:
            x = state[0]
            dx = (2.0 / tau_safe) * (u - x)
            return ca.vertcat(dx)
        else:
            x1, x2 = state[0], state[1]
            dx1 = x2
            dx2 = -(12.0 / (tau_safe**2)) * x1 - (6.0 / tau_safe) * x2 + (12.0 / (tau_safe**2)) * u
            return ca.vertcat(dx1, dx2)

    def observation(self, state, theta, u):
        tau = theta[0]
        tau_safe = ca.fmax(tau, 1e-6)
        if self.order == 1:
            x = state[0]
            return 2*u - x
        else:
            x2 = state[1]
            return u - tau_safe * x2
        
    
class LateralCarDynamic(ODESystem):
    def __init__(self, wheelbase):
        super().__init__(2, 4, 2)
        self.wheelbase = wheelbase

    def get_derivative(self, state, theta, u):
        vy, wz = state[0], state[1]
        a0, a1, b0, b1 = theta[0], theta[1], theta[2], theta[3]
        steering, vx = u[0], u[1]
        GR = 10
        alpha_f = np.atan2(vy + self.wheelbase * wz, vx)
        alpha_r = np.atan2(vy , vx) #(vy )/ vx
        rwa = steering/GR 
        vy_dot = a0 * (rwa - alpha_f) +  a1 * alpha_r - vx * wz 
        wz_dot = b0 * (rwa - alpha_f) +  b1 * alpha_r
        f = vertcat(vy_dot, wz_dot)
        return f
    
    def get_input_signals(self, t):
        w = 0.7
        u = 0.04 * jnp.cos(t*0.25*w) * jnp.sin(w*t) 
        return [u, 10.0]

    def observation(self, state, theta, u):
        return state

class LotkaVoltera(ODESystem):
    def __init__(self):
        super().__init__(2, 4, 0)
 
    def get_derivative(self, state, theta, u):
        x, y = state[0], state[1]
        alpha, beta, gamma, delta = theta[0], theta[1], theta[2], theta[3]
        
        dx_dt = alpha * x - (beta) * x * y #+ u[0]
        dy_dt = delta * x * y - gamma * y
        f = vertcat(dx_dt, dy_dt)
        return f
    
    def get_input_signals(self, t):
        return []
 
    def observation(self, state, theta, u):
        x, y = state[0], state[1]
        observed = vertcat(x, y)
        return observed
    
class Attractor(ODESystem):
    def __init__(self):
        super().__init__(3, 3, 0)

    def get_derivative(self, state, theta, u):
        x, y, z =  state[0], state[1], state[2]
        alpha, beta, gamma = theta[0], theta[1], theta[2]
        dx_dt = alpha * (y-x)
        dy_dt =  x * (beta - z) - y
        dz_dt =  x * y - gamma * z
        f = vertcat(dx_dt, dy_dt, dz_dt)
        return f
    
    def observation(self, state, theta, u):
        x, y, z = state[0], state[1], state[2]
        #observed = vertcat(np.sqrt(x**2 + y**2 + ), x/y)
        observed = vertcat(x,y, z)
        return observed



class OscillatorModel(ODESystem):
    def __init__(self):
        super().__init__(nx=2, nu=0, np=2)

   
    def get_derivative(self, state, params, input_signals):
        x1, x2 = state[0], state[1]
        omega, zeta = params[0], params[1]
        dx1 = x2
        dx2 = -omega**2 * x1 - 2*zeta*omega * x2
        return vertcat(dx1, dx2)
    
    def observation(self, state, theta, u):
        return state

class MassSpringDamper(ODESystem):
    def __init__(self, m = 1):
        self.mass = m
        super().__init__(nx=2, nu=1, np=2)

   
    def get_derivative(self, state, params, input_signals):
        x1, x2 = state[0], state[1]
        k, c = params[0], params[1]
        u = input_signals[0] if input_signals.shape[0] > 0 else 0.0

        dx1 = x2
        dx2 = (u - k*x1 - c*x2) / self.mass
        return vertcat(dx1, dx2)  
    
    
    def get_input_signals(self, t):
        import math
        w = 0.7
        u = 0.8 * math.cos(t * 0.25 * w) * math.sin(w * t)
        if t < 1:
            u = 0
        return [u]
   
class KinematicBycicle(ODESystem):
    def __init__(self, wheelbase):
        super().__init__(nx=1, nu=2, np=2)
        self.wheelbase = wheelbase

    def get_derivative(self, state, params, input_signals):
        psi = state[0]
        GR = params[0]
        offset = params[1]
        vx = input_signals[0]
        steering = input_signals[1]
        rwa = GR * steering + offset
        dpsi = vx * ca.tan(rwa) / self.wheelbase
        return ca.vertcat(dpsi)


class KinematicBycicleActuator(ODESystem):
    def __init__(self, wheelbase, kp = 80.9, kv = 80.61):
        super().__init__(nx=3, nu=2, np=2)
        self.wheelbase = wheelbase
        self.kp = kp
        self.kv = kv

    def get_derivative(self, state, params, input_signals):
        psi = state[0]
        delta = state[2]
        delta_dot = state[2]
        GR = params[0]
        offset = params[1]
        vx = input_signals[0]
        steering = input_signals[1]
        dpsi = vx * (np.tan(delta)) / (self.wheelbase) 
        ddelta = delta_dot
        rwa = GR * steering + offset
        ddelta_dot = self.kp * (rwa - delta) - self.kv * delta_dot
        return ca.vertcat(dpsi, ddelta, ddelta_dot)

    def observation(self, state, theta, u):
        # измеряем только курс
        return state[0]



class RosenzweigMacArthur(ODESystem):
    def __init__(self):
        super().__init__(nx=2, nu=0, np=1)

   
    def get_derivative(self, state, params, input_signals):
        x, y = state[0], state[1]
        r, K, a, h, e, m = params[0], params[1], params[2], params[3], params[4], params[5]
        
        # Правые части
        dx = r * x * (1 - x/K) - (a * x**2 * y) / (1 + a * h * x**2)
        dy = e * (a * x**2 * y) / (1 + a * h * x**2) - m * y
        rhs = vertcat(dx, dy)
        return rhs
    

class Quadrotor2D(ODESystem):
    """
    Упрощенная 2D-модель квадрокоптера (движение в плоскости X-Z).

    Состояние (state): [x, z, phi, x_dot, z_dot, phi_dot]
        x, z     - позиция в горизонтальной и вертикальной плоскостях,
        phi      - угол тангажа (pitch),
        x_dot    - горизонтальная скорость,
        z_dot    - вертикальная скорость,
        phi_dot  - угловая скорость.

    Параметры (params): [m, J, g, k_T, k_M]
        m  - масса дрона,
        J  - момент инерции,
        g  - ускорение свободного падения,
        k_T - коэффициент тяги,
        k_M - коэффициент момента (связь углового ускорения с силой).

    Входной сигнал (input_signals): [u]
        u - разность сил тяги между левым и правым роторами (управляющий сигнал).
    """

    def __init__(self):
        super().__init__(nx=6, nu=1, np=1)

    def get_derivative(self, state, params, input_signals):
        # Распаковка состояния
        x, z, phi, x_dot, z_dot, phi_dot = state[0], state[1], state[2], state[3], state[4], state[5]

        # Распаковка параметров
        #m, J, k_T, k_M = params[0], params[1], params[2], params[3]
        k_T = params[0]
        J, m, k_M = 10, 1, 1
        # Управляющий сигнал
        u = input_signals[0]
        g = 10
        # Сила тяги, создаваемая роторами (упрощённо, без динамики моторов)
        F_total = k_T * u

        # Вычисление производных
        dx_dt = x_dot
        dz_dt = z_dot
        dphi_dt = phi_dot

        # Ускорения: горизонтальное, вертикальное и угловое
        dx_dot_dt = -(F_total / m) * np.sin(phi)
        dz_dot_dt = (F_total / m) * np.cos(phi) - g
        dphi_dot_dt = (k_M / J) * u

        # Собираем вектор производных
        rhs = vertcat(dx_dt, dz_dt, dphi_dt, dx_dot_dt, dz_dot_dt, dphi_dot_dt)

        return rhs
    
    def get_input_signals(self, t):
        u = 0.5*np.sin(0.05*t) + t
        return [u]


class Integrator(ODESystem):

    def __init__(self):
        super().__init__(nx=2, nu=1, np=1)

    def get_derivative(self, state, params, input_signals):
        # Распаковка состояния
        x, v = state[0], state[1]

        # Распаковка параметров
        #m, J, k_T, k_M = params[0], params[1], params[2], params[3]
        k = params[0]
       
        dx_dt = v

        u = input_signals[0]
        # Ускорения: горизонтальное, вертикальное и угловое
        dv_dt = k*u


        # Собираем вектор производных
        rhs = vertcat(dx_dt, dv_dt)

        return rhs

    def get_input_signals(self, t):
        u = 0.5*np.sin(0.5*t) + t*0.001*np.sin(t)
        return [u]
    
    def observation(self, state, theta, u):
        return state[0]