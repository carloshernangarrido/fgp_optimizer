import numpy as np
import scipy as sp


def plot_results(axs, t, f_reaction, impulse, d, v, t_load=None, f_load=None, label=''):
    axs[0].set_ylabel('force')
    if t_load is not None:
        axs[0].plot(t_load, f_load, label='load', color='black')
        axs[1].plot(t, np.hstack((0, sp.integrate.cumtrapz(f_load, x=t_load))), label='load impulse', color='black')
    axs[0].plot(t, f_reaction, label=label+' reaction')
    axs[0].legend()
    axs[1].set_ylabel('impulse')
    axs[1].plot(t, impulse, label=label+' impulse')
    axs[2].set_ylabel('displacement')
    axs[2].plot(t, d)
    axs[3].set_ylabel('velocities')
    axs[3].plot(t, v)
    axs[3].set_xlabel('time')

