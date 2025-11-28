from matplotlib.gridspec import GridSpec
import numpy as np
from gauss_newton_math import TimeIntervalManager, MultipleShooting
import matplotlib.pyplot as plt

def plot_ax(ax, data, use3d, color = None, label = None, scatter = False):
    #use3d = data.shape[1] > 2
    fontsize = 20
    if(scatter):
        if(use3d):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = color, marker = "x", label = label)     
        else:
            ax.scatter(data[:, 0], data[:, 1], c = color, marker = "x", label = label)     
    else:

        if(use3d):
            ax.plot(data[:, 0], data[:, 1],  data[:, 2], c = color, label = label)
        else:
            ax.plot(data[:, 0], data[:, 1], c = color, label = label)
        
    ax.legend(fontsize = fontsize)


def plot_solution(fig, problem: MultipleShooting,  THETA_HIST:  np.array, validation = False, plot_xy = False, plot_theta = True, plot_true_soution = False, index = -1, theta_true = None):    
    fontsize = 20
    k = len(problem.t_eval_measurements_batches)
    N = 1
    if(plot_xy):
        N +=2
    if(plot_theta):
        N +=2
        
    gs = GridSpec(N, k, figure=fig, hspace = 0.2, wspace = 0.1)

    use3d = problem.state_full_batches[0].shape[1] > 2
    if(use3d):
        ax1 = fig.add_subplot(gs[0:2, :], projection='3d')
        plt.setp(ax1.get_zticklabels(), fontsize=fontsize)
        ax1.set_zlabel('z', fontsize=fontsize)
    else:
        ax1 = fig.add_subplot(gs[0:2, :])

    plt.setp(ax1.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax1.get_yticklabels(), fontsize=fontsize)
    
    ax1.tick_params(axis='both', which='major', length=20)
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlabel('x', fontsize=fontsize)
    ax1.set_ylabel('y', fontsize=fontsize)
    if(plot_theta):
        ax2 = fig.add_subplot(gs[-1, :])

    axes_f = []
    axes_s = []
    if(plot_xy):
        for i in range(len(problem.state_measured_batches)):
            ax = fig.add_subplot(gs[2, i])
            ax.grid(True)
            ax.set_xlabel('time, s', fontsize=fontsize)
            ax.set_ylabel('x', fontsize=fontsize)
            plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize)

            axes_f.append(ax)
            
            ax = fig.add_subplot(gs[3, i])
            ax.set_xlabel('time, s', fontsize=fontsize)
            ax.set_ylabel('y', fontsize=fontsize)
            plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize)

            ax.grid(True)
            axes_s.append(ax)
    
    theta_full = THETA_HIST[index]

    if(plot_true_soution):
        t_eval_batches_debug, state_full_batches_debug = problem.full_trajectory

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
        if(plot_true_soution):
            plot_ax(ax1, state_full_batches_debug[batch], use3d,  color = 'grey', label = "True solution")
    
        for shoot in range(min(len(shoot_indexes) -1, 100)):
            #print(f'shoot {shoot}')
            if(validation):
                c0 = state_full[shoot_indexes[shoot]]
                time_manger = TimeIntervalManager(problem.N_shoot, t_eval_measurements)
                t_eval_curr,  _  = time_manger.get_time_interval(shoot)
            else:
                c0 = theta_full[THETA_LENGTH + (problem.N_shoot) * batch * STATE_LENGTH + shoot * STATE_LENGTH : THETA_LENGTH + (problem.N_shoot) * batch * STATE_LENGTH +  (shoot + 1) * STATE_LENGTH]
                t_eval_curr, _ = problem.get_time_interval(shoot, batch)
                

            solution = problem.system.get_solution_jax(c0, theta_full[:THETA_LENGTH], t_eval_curr)
            solution_debug = problem.system.get_solution_jax(c0, theta_full[:THETA_LENGTH], np.linspace(t_eval_curr[0], t_eval_curr[-1], 500))
            state_observed = np.zeros((solution.T.shape[0], STATE_LENGTH))
            state_observed_debug = np.zeros((solution_debug.T.shape[0], STATE_LENGTH))
            for i, state in enumerate(solution.T):
                state_observed[i] = state  #system.h_x(state,  t_eval_curr[i], theta_full[:THETA_LENGTH])
            for i, state in enumerate(solution_debug.T):
                state_observed_debug[i] = state  #system.h_x(state,  t_eval_curr[i], theta_full[:THETA_LENGTH])
            
                
            if(len(axes_f) > 0):
                axes_f[batch].plot(np.linspace(t_eval_curr[0], t_eval_curr[-1], 500), state_observed_debug[:, 0])
            if(state_observed_debug.shape[1] > 1):
                plot_ax(ax1, state_observed_debug, use3d)
                if(len(axes_s) > 0):
                    axes_s[batch].plot(np.linspace(t_eval_curr[0], t_eval_curr[-1], 500), state_observed_debug[:, 1])
        

        if(len(axes_f) > 0):
            axes_f[batch].scatter(t_eval_measurements, state_full[:, 0],  color='green',  marker='x',  s = 10)
            
        if(state_full.shape[1] > 1):
            plot_ax(ax1, state_full, use3d, color='green', label = "Measurements", scatter = True)
            if(len(axes_s) > 0):
                axes_s[batch].scatter(t_eval_measurements, state_full[:, 1],  color='green',  marker='x',  s = 10)


    # plt.tight_layout()
    if(plot_theta):
        plt.setp(ax2.get_xticklabels(), fontsize=fontsize)
        plt.setp(ax2.get_yticklabels(), fontsize=fontsize)
        data = np.array(THETA_HIST)[:index, :THETA_LENGTH]
    
        ax2.set_xlabel('iter', fontsize=fontsize)
        ax2.set_ylabel('Estimated coefs', fontsize=fontsize)
    
        ax2.grid(True)

        colors_arr = ['blue', 'orange', 'green', 'black', 'red']
        for i in range(data.shape[1]):
            if(plot_true_soution):
                ax2.axhline(y=theta_true[i], linestyle='--', color=colors_arr[i], label=rf'$coef{i}$')
            ax2.step(range(len(data)), data[:, i], color=colors_arr[i], where='post')

        ax2.legend(fontsize = fontsize)
