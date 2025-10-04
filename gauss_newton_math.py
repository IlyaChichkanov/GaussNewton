import numpy as np
from scipy.integrate import solve_ivp
from systems import System

from jaxadi import convert
from jax import numpy as jnp
from jax.experimental.ode import odeint
from casadi import SX, vertcat, jacobian, Function
import casadi as ca
import time as tm_module
ATOL = 1e-5
RTOL = 1e-5

class SystemJacobian:
    def __init__(self, f_sym: System, method: str =  'RK45'):
        self.f_sym = f_sym
        self.method = method
        state_var, inp_signal_var, theta_var, f = self.f_sym()

        h_observ = self.f_sym.observation()
        self.inp_signal_len = len(inp_signal_var.elements())
        self.res_f = Function('func', [*state_var.elements(), *inp_signal_var.elements(), *theta_var.elements()], [f])
        self.res_f_jax = convert(self.res_f, compile = True) 

        self.res_h = Function('h_x', [*state_var.elements(), *inp_signal_var.elements(), *theta_var.elements()], [h_observ])
        #self.res_h_jax = convert(Function('h_x', [*state_var.elements(), *inp_signal_var.elements(), *theta_var.elements()], [h_observ]), compile = True) 

        J_h_x = jacobian(h_observ, vertcat(*[*state_var.elements()]))
        self.compute_jacobian_h_x = Function('J_h_x', [*state_var.elements(), *inp_signal_var.elements(), *theta_var.elements()], [J_h_x])
        #self.compute_jacobian_h_x_jax = convert(self.compute_jacobian_h_x, compile = True) 

        J_h_theta = jacobian(h_observ, vertcat(*[*theta_var.elements()]))
        self.compute_jacobian_h_theta = Function('J_h_theta', [*state_var.elements(), *inp_signal_var.elements(), *theta_var.elements()], [J_h_theta])
        #self.compute_jacobian_h_theta_jax =convert(self.compute_jacobian_h_theta, compile = True) 

        J_p = jacobian(f, vertcat(*[*theta_var.elements()]))
        self.compute_jacobian_theta = Function('J_p', [*state_var.elements(), *inp_signal_var.elements(), *theta_var.elements()], [J_p])
        self.compute_jacobian_theta_jax = convert(Function('J_p', [*state_var.elements(), *inp_signal_var.elements(), *theta_var.elements()], [J_p]), compile = True)

        J_x = jacobian(f, vertcat(*[*state_var.elements()]))
        self.compute_jacobian_x = Function('J_x', [*state_var.elements(), *inp_signal_var.elements(), *theta_var.elements()], [J_x])
        self.compute_jacobian_x_jax = convert(Function('J_x', [*state_var.elements(), *inp_signal_var.elements(), *theta_var.elements()], [J_x]), compile = True)

        self.STATE_LENGTH = len(state_var.elements())
        self.THETA_LENGTH = len(theta_var.elements())
        self.MEAS_LENGTH = len(h_observ.elements())

        
    def get_dimentions(self):
        return self.STATE_LENGTH, self.THETA_LENGTH, self.MEAS_LENGTH
    
    def get_inp_signals(self, t):
        try:
            inp_signals = self.f_sym.get_input_signals(t)
        except ValueError:
            inp_signals = np.zeros(self.inp_signal_len)
            print(f'df_dx interpolation error, time {t}')
        return inp_signals

    def h_x(self, state, t, theta):
        inp_signals = np.zeros(self.inp_signal_len) #self.get_inp_signals(t)
        return np.array(self.res_h(*[*state, *inp_signals, *theta])).T[0]
    
    # def h_x_jax(self, state, t, theta):
    #     inp_signals = self.get_inp_signals(t)
    #     return np.array(self.res_h_jax(*[*state, *inp_signals, *theta])).flatten()
    
    def f_x_theta(self, state, t, theta):
        inp_signals = self.get_inp_signals(t)
        return np.array(self.res_f(*[*state, *inp_signals, *theta])).T[0]
    
    def f_x_theta_jax(self, y, t, *theta):
        inp_signals = self.get_inp_signals(t)
        return jnp.array(self.res_f_jax(*y, *inp_signals, *theta)[0].flatten())
        
    def dh_dx(self, state, t, theta):
        inp_signals = np.zeros(self.inp_signal_len) #self.get_inp_signals(t)
        return np.array(self.compute_jacobian_h_x(*[*state, *inp_signals, *theta]))
    
    # def dh_dx_jax(self, state, t, theta):
    #     inp_signals = self.get_inp_signals(t)
    #     return np.array(self.compute_jacobian_h_x_jax(*[*state, *inp_signals, *theta])).squeeze()
    
    def dh_dtheta(self, state, t, theta):
        inp_signals = np.zeros(self.inp_signal_len)#self.get_inp_signals(t)
        return np.array(self.compute_jacobian_h_theta(*[*state, *inp_signals, *theta])).squeeze()

    # def dh_dtheta_jax(self, state, t, theta):
    #     inp_signals = self.get_inp_signals(t)
    #     return np.array(self.compute_jacobian_h_theta_jax(*[*state, *inp_signals, *theta]))
    
    def df_dtheta_jax(self, state, t, theta):
        inp_signals = self.get_inp_signals(t)
        return jnp.array(self.compute_jacobian_theta_jax(*[*state, *inp_signals, *theta]))[0]
      
    def df_dtheta(self, state, t, theta):
        inp_signals = self.get_inp_signals(t)
        return np.array(self.compute_jacobian_theta(*[*state, *inp_signals, *theta]))


    def df_dx(self, state, t, theta):
        inp_signals = self.get_inp_signals(t)
        return np.array(self.compute_jacobian_x(*[*state, *inp_signals, *theta]))
    
    def df_dx_jax(self, state, t, theta):
        inp_signals = self.get_inp_signals(t)
        return jnp.array(self.compute_jacobian_x_jax(*[*state, *inp_signals, *theta]))
    
    def get_jacobian_solution(self, c0, theta, t_eval1):
        J0 = np.concatenate((np.zeros((self.STATE_LENGTH, self.THETA_LENGTH)).flatten(), np.identity((self.STATE_LENGTH)).flatten()))
        initial_conditions = np.concatenate((c0, J0))
        system = lambda t, y: self.make_full_system(y, t, theta[:self.THETA_LENGTH])
        solution1 = solve_ivp(
            system,
            (t_eval1[0], t_eval1[-1]),
            initial_conditions,
            t_eval = t_eval1,
            method = self.method,
            atol = ATOL, rtol = RTOL
        )

        if(not (solution1.success)):
            raise ValueError("solution disconverge")
        return solution1.y

    def get_solution(self, c0, theta, t_eval):
        system = lambda t, y: self.f_x_theta(y, t, theta[:self.THETA_LENGTH])    #self.make_full_system(y, t, theta[:self.THETA_LENGTH])
        solution1 = solve_ivp(
            system,
            (t_eval[0], t_eval[-1]),
            c0,
            t_eval = t_eval,
            method = self.method,
            atol = ATOL, rtol = RTOL
        )

        if(not (solution1.success)):
            print(solution1.message)
            raise ValueError("solution disconverge")
        return solution1.y

    def get_jacobian_solution_jax(self, c0, theta, t_eval):
        J0 = jnp.concatenate((jnp.zeros((self.STATE_LENGTH, self.THETA_LENGTH)).flatten(), jnp.identity((self.STATE_LENGTH)).flatten()))
        initial_conditions = np.concatenate((jnp.array(c0), J0))
        solution = odeint(
            self.make_full_system_jax,
            jnp.array(initial_conditions),
            jnp.array(t_eval),
            *theta[:self.THETA_LENGTH]
        )
        return np.array(solution).T
    
    def get_solution_jax(self, c0, theta, t_eval):
        solution = odeint(
            self.f_x_theta_jax,
            jnp.array(c0),
            jnp.array(t_eval),
            *theta[:self.THETA_LENGTH]
        )
        return np.array(solution).T
          

    def JacobianX(self, state,t, theta):
        xy = state[:self.STATE_LENGTH]
        Jx = state[self.STATE_LENGTH : self.STATE_LENGTH + self.THETA_LENGTH * self.STATE_LENGTH].reshape((self.STATE_LENGTH, self.THETA_LENGTH))
        dJx = self.df_dx(xy, t, theta)@Jx + self.df_dtheta(xy, t, theta)
        return dJx.flatten()

    def JacobianC(self, state, t, theta):
        xy = state[:self.STATE_LENGTH]
        start_ind = self.STATE_LENGTH + self.THETA_LENGTH * self.STATE_LENGTH
        end_ind = start_ind + self.STATE_LENGTH * self.STATE_LENGTH
        Jc = state[start_ind: end_ind].reshape((self.STATE_LENGTH, self.STATE_LENGTH))
        dJc = self.df_dx(xy, t, theta)@Jc
        return dJc.flatten()

    def make_full_system(self, state, t, theta):
        xy = state[: self.STATE_LENGTH]
        dstate = np.concatenate((self.f_x_theta(xy, t, theta), 
                                 self.JacobianX(state, t, theta), self.JacobianC(state, t, theta)))
        return dstate

    def JacobianX_jax(self, state,t, theta):
        xy = state[:self.STATE_LENGTH]
        Jx = state[self.STATE_LENGTH : self.STATE_LENGTH + self.THETA_LENGTH * self.STATE_LENGTH].reshape((self.STATE_LENGTH, self.THETA_LENGTH))
        dJx = self.df_dx_jax(xy, t, theta)@Jx + self.df_dtheta_jax(xy, t, theta)
        return dJx.flatten()

    def JacobianC_jax(self, state, t, theta):
        xy = state[:self.STATE_LENGTH]
        start_ind = self.STATE_LENGTH + self.THETA_LENGTH * self.STATE_LENGTH
        end_ind = start_ind + self.STATE_LENGTH * self.STATE_LENGTH
        Jc = state[start_ind: end_ind].reshape((self.STATE_LENGTH, self.STATE_LENGTH))
        dJc = self.df_dx_jax(xy, t, theta)@Jc
        return dJc.flatten()

    def make_full_system_jax(self, state, t, *theta):
        xy = state[: self.STATE_LENGTH]
        dstate = jnp.concatenate((self.f_x_theta_jax(xy, t, *theta), 
                                 self.JacobianX_jax(state, t, theta), self.JacobianC_jax(state, t, theta)))
        return dstate

    

class TimeIntervalManager:
    def __init__(self, N_shoot, t_eval_measurements):
        self.t_eval_measurements = t_eval_measurements
        N_measurement = len(t_eval_measurements)
        self.measurement_indexes = np.arange(0, N_measurement, 1, dtype=int )
        shoot_indexes = self.measurement_indexes[0:-1:int(len(self.measurement_indexes)/N_shoot)]
        self.shoot_indexes = np.append(shoot_indexes, self.measurement_indexes[-1])
        self.N_shoot = len(self.shoot_indexes) - 1

    def get_time_interval(self, shoot):
        measurement_indexes_curr = self.measurement_indexes[np.where((self.measurement_indexes >= self.shoot_indexes[shoot]) * (self.measurement_indexes < self.shoot_indexes[shoot + 1]))]
        t_eval_curr_measur = []
        for id in measurement_indexes_curr:
            t_eval_curr_measur.append(self.t_eval_measurements[id])
        t0 = t_eval_curr_measur[0]
        t1 = self.t_eval_measurements[measurement_indexes_curr[-1] + 1]
        t_eval_curr_measur.append(t1)
        t_eval_curr_measur = np.array(t_eval_curr_measur)

        return t_eval_curr_measur, measurement_indexes_curr
    

class MultipleShooting:
    def __init__(self, system: SystemJacobian, N_shoot: int, gamma: np.array, use_jax = False):
        self.system = system
        self.N_shoot = N_shoot
        self.gamma = gamma
        self.state_measured_batches = []
        self.state_full_batches = []
        self.t_eval_measurements_batches = []
        self.use_jax = use_jax

    def add_batch(self, state_full, state_measured, t_eval_measurements):
        self.state_measured_batches.append(state_measured)
        self.state_full_batches.append(state_full)
        self.t_eval_measurements_batches.append(t_eval_measurements)
        
    def get_time_interval(self, shoot, batch):
        time_manger = TimeIntervalManager(self.N_shoot, self.t_eval_measurements_batches[batch])
        return time_manger.get_time_interval(shoot)
    
    def make_full_theta(self, theta0):
        theta_full = np.copy(theta0)
        for state_measured, state_full, t_eval_measurements in zip(self.state_measured_batches, self.state_full_batches, self.t_eval_measurements_batches):
            N_measurement = len(t_eval_measurements)
            measurement_indexes = np.arange(0, N_measurement, 1, dtype=int )
            shoot_indexes = measurement_indexes[0:-1:int(len(measurement_indexes)/self.N_shoot)]
            shoot_indexes = np.append(shoot_indexes, measurement_indexes[-1])
            for i in range(len(shoot_indexes)-1):
                id = shoot_indexes[i]
                if(1):
                    c0_ = state_full[id]
                theta_full = np.concatenate((theta_full, c0_ ))
        return theta_full

    def concantenate_jacobian(self, J1, J2):
        STATE_LENGTH, THETA_LENGTH, MEAS_LEN = self.system.get_dimentions()
        J2_theta = J2[:, :THETA_LENGTH]
        J2_c0 = J2[:, THETA_LENGTH:]
        zeros1 = np.zeros((J1.shape[0], J2_c0.shape[1]))#
        zeros2 = np.zeros((J2_theta.shape[0], J1.shape[1] - THETA_LENGTH))#
        #zeros1 = np.zeros_like(J2_c0)
        return np.block([[J1, zeros1], [J2_theta, zeros2, J2_c0]])
    
    def solve(self, theta_full):
        J = J_G = R = R_G = []
        for batch, (state_measured, t_eval_measurements) in enumerate(zip(self.state_measured_batches, self.t_eval_measurements_batches)):
            print(batch, "solve batch {batch}")
            J_, J_G_, R_, R_G_ = self.solve_batch(theta_full, state_measured, t_eval_measurements, batch)
            if(len(J) == 0):
                J = J_
                J_G = J_G_
                R = R_
                R_G = R_G_
            else:
                J = self.concantenate_jacobian(J, J_)
                J_G = self.concantenate_jacobian(J_G, J_G_)
                R = np.hstack((R, R_))
                R_G = np.hstack((R_G, R_G_))


        return J, R, J_G, R_G


    def solve_batch(self, theta_full, state_measured, t_eval_measurements, batch):
        STATE_LENGTH, THETA_LENGTH, MEAS_LEN = self.system.get_dimentions()
        INDEX_THETA = range(0, STATE_LENGTH * THETA_LENGTH)
        INDEX_C = range(STATE_LENGTH * THETA_LENGTH, STATE_LENGTH * (THETA_LENGTH + STATE_LENGTH))
        N_measurement = state_measured.shape[0]
    
        time_manger = TimeIntervalManager(self.N_shoot, t_eval_measurements)
        N_shoot = time_manger.N_shoot
        J = np.zeros((N_measurement, MEAS_LEN, THETA_LENGTH + N_shoot * STATE_LENGTH))
        J_G = np.zeros((N_shoot-1, STATE_LENGTH, THETA_LENGTH + (N_shoot) * STATE_LENGTH))
        R = np.zeros((N_measurement, MEAS_LEN))
        R_G = np.zeros((N_shoot - 1, STATE_LENGTH))
        ind = 0
        Jx_prev = None
        Jc_prev = None
        state_prev = None
        for shoot in range(N_shoot):
            c0 = theta_full[THETA_LENGTH + N_shoot * batch * STATE_LENGTH  + shoot * STATE_LENGTH: THETA_LENGTH + N_shoot * batch * STATE_LENGTH + (shoot + 1) * STATE_LENGTH]
            t_eval_curr_measur, measurement_indexes_curr = time_manger.get_time_interval(shoot)
            start = tm_module.time()
            if(self.use_jax):
                solution = self.system.get_jacobian_solution_jax(c0, theta_full[:THETA_LENGTH], t_eval_curr_measur)
            else:
                solution = self.system.get_jacobian_solution(c0, theta_full[:THETA_LENGTH], t_eval_curr_measur)
            time_finish = tm_module.time() - start
            #print("time_finish1",time_finish)
            start = tm_module.time()
            J_raw = solution[STATE_LENGTH:]
            state_sample = solution[0:STATE_LENGTH]
            for i in range(len(measurement_indexes_curr)):
                state = state_sample[:, i]
                t = t_eval_curr_measur[i]
                dh_dx =  self.system.dh_dx(state, t, theta_full[:THETA_LENGTH])
                dh_dtheta =  self.system.dh_dtheta(state, t, theta_full[:THETA_LENGTH])
                dx_dtheta = J_raw[INDEX_THETA, i].reshape(STATE_LENGTH, THETA_LENGTH)
                dx_dc = J_raw[INDEX_C, i].reshape(STATE_LENGTH, STATE_LENGTH)
                J_x = dh_dx@dx_dtheta + dh_dtheta
                J_c = dh_dx@dx_dc
                J[ind][:, :THETA_LENGTH] = J_x /len(measurement_indexes_curr)
                J[ind][:, THETA_LENGTH + STATE_LENGTH * shoot : THETA_LENGTH + (STATE_LENGTH) * (shoot + 1)] = J_c/len(measurement_indexes_curr)
                d = state_measured[ind] - self.system.h_x(state, t, theta_full[:THETA_LENGTH]) 
                d*= self.gamma
                R[ind]=  np.array(d) / len(measurement_indexes_curr)
                ind +=1

            if(shoot > 0):
                J_G[shoot - 1][:, :THETA_LENGTH] = Jx_prev
                J_G[shoot - 1][:, THETA_LENGTH + STATE_LENGTH * (shoot -1) : THETA_LENGTH + STATE_LENGTH * (shoot)] = Jc_prev
                J_G[shoot - 1][:, THETA_LENGTH + STATE_LENGTH * (shoot) : THETA_LENGTH + (STATE_LENGTH) * (shoot + 1)] = - np.eye(STATE_LENGTH)
                R_G[shoot - 1] = -(state_prev - c0)

            
            time_finish = tm_module.time() - start
            #print("time_finish2",time_finish)
            Jx_prev = J_raw[INDEX_THETA, -1].reshape(STATE_LENGTH, THETA_LENGTH)
            Jc_prev = J_raw[INDEX_C, -1].reshape(STATE_LENGTH, STATE_LENGTH) 
            state_raw = state_sample[:STATE_LENGTH]
            state_prev = state_raw[:, -1] 

        J = J.reshape(N_measurement * MEAS_LEN, -1)
        J_G = J_G.reshape((N_shoot-1) * STATE_LENGTH, -1)
        R = R.reshape(N_measurement * MEAS_LEN)
        R_G = R_G.reshape((N_shoot -1) * STATE_LENGTH)

        return J, J_G, R, R_G
    
class Regressor:
    def __init__(self, system: System):
        self.system = system
        inp_signal_var, theta_var, h_observ = system.observation()
        self.inp_signal_len = len(inp_signal_var.elements())
        self.theta_len = len(theta_var.elements())
        self.meas_len = len(h_observ.elements())
        self.res_h = Function('h_x', [*inp_signal_var.elements(), *theta_var.elements()], [h_observ])
        
        J_h_theta = jacobian(h_observ, theta_var)
        self.compute_jacobian_h_theta = Function('J_h_x', [*inp_signal_var.elements(), *theta_var.elements()], [J_h_theta])

    def get_inp_signals(self, t):
        try:
            inp_signals = self.f_sym.get_input_signals(t)
        except ValueError:
            inp_signals = np.zeros(self.inp_signal_len)
            print(f'df_dx interpolation error, time {t}')
        return inp_signals
    
    def h_x(self, t, theta):
        inp_signals = np.zeros(self.inp_signal_len) #self.get_inp_signals(t)
        return np.array(self.res_h(*[*inp_signals, *theta])).T[0]
    
    def dh_dtheta(self, t, theta):
        inp_signals = np.zeros(self.inp_signal_len) #self.get_inp_signals(t)
        return np.array(self.compute_jacobian_h_theta(*[*inp_signals, *theta]))
    
    def solve(self, theta, state_measured, t_eval_measurements):
        N_measurement = len(t_eval_measurements)
        R = np.zeros((N_measurement, self.meas_len))
        J = np.zeros((len(t_eval_measurements), self.meas_len, self.theta_len))
        for ind, t in enumerate(t_eval_measurements):
            J[ind] = self.dh_dtheta(t, theta) 
            R[ind] = state_measured[ind] - self.h_x(t, theta)

        J = J.reshape(N_measurement * self.meas_len, -1)
        R = R.reshape(N_measurement * self.meas_len)
        return J, R
    
