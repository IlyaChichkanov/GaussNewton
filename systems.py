import casadi as ca
from casadi import SX, vertcat
import numpy as np
from abc import ABC, abstractmethod
import time
import jax
from jax import jit
from jax import numpy as jnp

class System:
    def __init__(self):
        pass

    @abstractmethod
    def get_system(self):
        pass

    @abstractmethod
    def __call__(self):
        return self.get_system()
    
    def observation(self):
        observed = vertcat(*self.state)
        return observed
    
    def get_input_signals(self, t):
        return []

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




class Lateral_car_dynamic(System):
    def __init__(self, wheelbase = 2.65):
        super().__init__()
        self.state = SX.sym('vy'), SX.sym('wz')
        self.wheelbase = wheelbase

    def get_system(self):
        vy, wz = self.state
        delta, vx = SX.sym('delta'), SX.sym('vx')
        a0, a1, b0, b1 = SX.sym('a0'), SX.sym('a1'), SX.sym('b0'), SX.sym('b1')
        GR = SX.sym('GR')
        alpha_f = np.atan2(vy  + self.wheelbase * wz, vx)
        alpha_r = np.atan2(vy , vx) #(vy )/ vx
        GR = 1
        vy_dot = a0 * (delta*GR - alpha_f) +  a1 * alpha_r - vx * wz 
        wz_dot = b0 * (delta*GR - alpha_f) +  b1 * alpha_r

        f = vertcat(vy_dot, wz_dot)
        return vertcat(vy, wz ), vertcat(delta, vx), vertcat(a0, a1, b0, b1), f
    

    # def get_system(self):
        
    #     vy, wz = self.state
    #     delta, vx = SX.sym('delta'), SX.sym('vx')
    #     a0, a1, b0, b1 = SX.sym('a0'), SX.sym('a1'), SX.sym('b0'), SX.sym('b1')
    #     GR = SX.sym('GR')
    #     alpha_f = np.atan2(vy  + self.wheelbase * wz, vx)
    #     alpha_r = np.atan2(vy , vx) #(vy )/ vx

    #     vy_dot = a0 * (delta*GR - alpha_f) +  a1 * alpha_r - vx * wz 
    #     wz_dot = b0 * (delta*GR - alpha_f) +  b1 * alpha_r

    #     f = vertcat(vy_dot, wz_dot)
    #     return vertcat(vy, wz ), vertcat(delta, vx), vertcat(a0, a1, b0, b1, GR), f
    
    # def observation(self):
    #     vy, wz = self.state
    #     observed = vertcat(wz)
    #     return observed
    
# class DynamicCar(System):
#     def __init__(self):
#         super().__init__()
#         self.state = SX.sym('vy'), SX.sym('wz'),  SX.sym('yaw') 
#         self.theta = SX.sym('a0'), SX.sym('a1'), SX.sym('b0'), SX.sym('b1'), SX.sym('GR1'), SX.sym('GR2')

#     def get_system(self):
#         wheelbase = 2.65
#         vy, wz, yaw = self.state
#         delta, vx = SX.sym('delta'), SX.sym('vx')
#         a0, a1, b0, b1,GR1, GR2 = self.theta

#         alpha_f = (vy  + wheelbase * wz)/ vx
#         alpha_r = (vy )/ vx

#         vy_dot = a0 * (delta - alpha_f) +  a1 * alpha_r - vx * wz 
#         wz_dot = b0 * (delta - alpha_f) +  b1 * alpha_r
        
#         vy_dot = GR1 * delta - a0* alpha_f +  a1 * alpha_r - vx * wz 
#         wz_dot = GR2 * delta - b0 * alpha_f +  b1 * alpha_r
#         psi_dot = wz

#         f = vertcat(vy_dot, wz_dot, psi_dot)
#         return  vertcat(*self.state), vertcat(delta, vx), vertcat(*self.theta), f
    
#     def observation(self):
#         vy, wz, yaw = self.state
#         observed = vertcat(wz, yaw)
#         return observed
    
class Lotka_voltera(System):
    def __init__(self):
        super().__init__()
        self.state = [SX.sym('x'), SX.sym('y')]
        self.theta = SX.sym('alpha'), SX.sym('beta'), SX.sym('gamma'), SX.sym('delta')
        self.u = [SX.sym('u')]
        
    def get_system(self):
        x, y = self.state
        alpha, beta, gamma, delta = self.theta
        
        dx_dt = alpha * x - (beta) * x * y + self.u[0]
        dy_dt = delta * x * y - gamma * y
        f = vertcat(dx_dt, dy_dt)
        return vertcat(*self.state), vertcat(*self.u), vertcat(*self.theta), f
    
    # def get_input_signals(self, t):
    #     return [0.1 * t - np.sin(t)]
    
    def get_input_signals(self, t):
        return [0.0]
    
    # def observation(self):
    #     x, y = self.state
    #     observed = vertcat(x + y, x - y)
    #     return vertcat(*self.state), observed

    # def observation(self):
    #     x, y = self.state
    #     observed = vertcat(np.sqrt(x**2 + y**2), x/y)
    #     return [x, y], observed
    
    def observation(self):
        x, y = self.state
        observed = vertcat(x , y)
        return observed
    
class Attractor(System):
    def __init__(self):
        self.state = SX.sym('x'), SX.sym('y'), SX.sym('z')
        
    def get_system(self):
        x, y, z = self.state
        alpha, beta, gamma = SX.sym('alpha'), SX.sym('beta'), SX.sym('gamma')
        dx_dt = alpha * (x - y)
        dy_dt =  x * (beta - z) - y
        dz_dt =  x * y - gamma * z
        f = vertcat(dx_dt, dy_dt, dz_dt)
        return vertcat(x, y, z), [],vertcat(alpha, beta, gamma), f
    
    def observation(self):
        x, y, z = self.state
        #observed = vertcat(np.sqrt(x**2 + y**2 + ), x/y)
        observed = vertcat(x, z)
        return observed



class LateralSemiDynamic(System):
    def __init__(self):
        self.state = SX.sym('psi'), SX.sym('w'), SX.sym('y')
        
    def get_system(self):
        psi, w, y = self.state
        T, D = SX.sym('T'), SX.sym('D')#, SX.sym('udersteer_gain')
        delta = SX.sym('delta')
        K = 2
        psi_dot = w
        w_dot = y
        y_dot = T * (K * delta - w) - D * y
        f = vertcat(psi_dot, w_dot, y_dot)
        return [psi, w, y], [delta], [T, D ], f
    
    def get_input_signals(self, t):
        return [(0.1 + 0.09 * t) * np.sin(0.9 *t)]
    
    def observation(self):
        psi, w, y = self.state
        observed = vertcat(psi, w)
        return observed


  