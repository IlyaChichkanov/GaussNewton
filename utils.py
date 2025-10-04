from matplotlib.gridspec import GridSpec
import numpy as np
from gauss_newton_math import TimeIntervalManager, MultipleShooting
import matplotlib.pyplot as plt

def plot_solution(problem: MultipleShooting, theta_full: np.array, THETA_HIST, validation = False):    
    k = len(problem.t_eval_measurements_batches)
    fig = plt.figure(figsize=(30, 15))
    gs = GridSpec(4, k, figure=fig, hspace = 0.2, wspace = 0.1)
    if(1):
        ax1 = fig.add_subplot(gs[0, :])
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

    ax2 = fig.add_subplot(gs[3, :])
    axes_f = []
    axes_s = []
    for i in range(len(problem.state_measured_batches)):
        ax = fig.add_subplot(gs[1, i])
        ax.grid(True)
        ax.set_xlabel('time, s')
        ax.set_ylabel('x')
        axes_f.append(ax)
        
        ax = fig.add_subplot(gs[2, i])
        ax.set_xlabel('time, s')
        ax.set_ylabel('y')
        ax.grid(True)
        axes_s.append(ax)
        
    for batch, (state_full, state_measured, t_eval_measurements) in \
                                        enumerate(zip(problem.state_full_batches, 
                                                      problem.state_measured_batches, 
                                                      problem.t_eval_measurements_batches)):
        print(f'batch {batch}')
        N_measurement = len(t_eval_measurements)
        measurement_indexes = np.arange(0, N_measurement, 1, dtype=int )
        shoot_indexes = measurement_indexes[0:-1:int(len(measurement_indexes)/problem.N_shoot)]
        shoot_indexes = np.append(shoot_indexes, measurement_indexes[-1])
        STATE_LENGTH, THETA_LENGTH, MEAS_LEN = problem.system.get_dimentions()
        for shoot in range(min(len(shoot_indexes) -1, 100)):
            print(f'shoot {shoot}')
            if(validation):
                c0 = state_full[shoot_indexes[shoot]]
                time_manger = TimeIntervalManager(problem.N_shoot, t_eval_measurements)
                t_eval_curr,  _  = time_manger.get_time_interval(shoot)
            else:
                c0 = theta_full[THETA_LENGTH + (problem.N_shoot) * batch * STATE_LENGTH + shoot * STATE_LENGTH : THETA_LENGTH + (problem.N_shoot) * batch * STATE_LENGTH +  (shoot + 1) * STATE_LENGTH]
                t_eval_curr, _ = problem.get_time_interval(shoot, batch)
                

            solution = problem.system.get_solution_jax(c0, theta_full[:THETA_LENGTH], t_eval_curr)
            state_observed = np.zeros((solution.T.shape[0], 2))
            for i, state in enumerate(solution.T):
                state_observed[i] = state  #system.h_x(state,  t_eval_curr[i], theta_full[:THETA_LENGTH])

            axes_f[batch].plot(t_eval_curr, state_observed[:, 0])
            if(state_observed.shape[1] > 1):
                ax1.plot(state_observed[:, 0], state_observed[:, 1])
                axes_s[batch].plot(t_eval_curr, state_observed[:, 1])
        

        axes_f[batch].scatter(t_eval_measurements, state_measured[:, 0],  color='green',  marker='x',  s=1)
        if(state_measured.shape[1] > 1):
            ax1.scatter(state_measured[:, 0] , state_measured[:, 1], color='green',  marker='x')
            axes_s[batch].scatter(t_eval_measurements, state_measured[:, 1],  color='green',  marker='x')

    plt.tight_layout()
    data = np.array(THETA_HIST)[:, :THETA_LENGTH]
    ax2.plot(range(len(data)), data)
    ax2.set_xlabel('iter')
    ax2.legend()
    ax2.grid(True)
    plt.show()