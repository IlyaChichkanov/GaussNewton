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

    def observation(self, state):
        return state

class Lotka_voltera(ODESystem):
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
 
    def observation(self, state):
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
    
    def observation(self, state):
        x, y, z = state[0], state[1], state[2]
        #observed = vertcat(np.sqrt(x**2 + y**2 + ), x/y)
        observed = vertcat(x,y, z)
        return observed



# class LateralSemiDynamic(System):
#     def __init__(self):
#         self.state = SX.sym('psi'), SX.sym('w'), SX.sym('y')
        
#     def get_system(self):
#         psi, w, y = self.state
#         T, D = SX.sym('T'), SX.sym('D')#, SX.sym('udersteer_gain')
#         delta = SX.sym('delta')
#         K = 2
#         psi_dot = w
#         w_dot = y
#         y_dot = T * (K * delta - w) - D * y
#         f = vertcat(psi_dot, w_dot, y_dot)
#         return [psi, w, y], [delta], [T, D ], f
    
#     def get_input_signals(self, t):
#         return [(0.1 + 0.09 * t) * np.sin(0.9 *t)]
    
#     def observation(self):
#         psi, w, y = self.state
#         observed = vertcat(psi, w)
#         return observed


class OscillatorModel(ODESystem):
    def __init__(self):
        super().__init__(nx=2, nu=0, np=2)

   
    def get_derivative(self, state, params, input_signals):
        x1, x2 = state[0], state[1]
        omega, zeta = params[0], params[1]
        dx1 = x2
        dx2 = -omega**2 * x1 - 2*zeta*omega * x2
        return vertcat(dx1, dx2)
    
    def observation(self, state):
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
    
    def observation(self, state):
        return state
    
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

    def observation(self, state):
        return state[0]

    def get_input_signals(self, t):
        import math
        w = 0.7
        steering = 0.8 * math.cos(t * 0.25 * w) * math.sin(w * t)
        if t < 25:
            steering = 0
        return [10.0, steering]

class KinematicBycicleActuator(ODESystem):
    def __init__(self, wheelbase):
        super().__init__(nx=3, nu=2, np=2)
        self.wheelbase = wheelbase

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
        kp = 80.9
        kv = 80.61
        rwa = GR * steering + offset
        ddelta_dot = kp * (rwa - delta) - kv * delta_dot
        return ca.vertcat(dpsi, ddelta, ddelta_dot)

    def observation(self, state):
        # измеряем только курс
        return state[0]

    def get_input_signals(self, t):
        import math
        w = 0.7
        steering = 0.8 * math.cos(t * 0.25 * w) * math.sin(w * t)
        if t < 25:
            steering = 0
        return [10.0, steering]

