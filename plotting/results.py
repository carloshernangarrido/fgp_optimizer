import numpy as np
import scipy as sp


def plot_results(axs, t, f_reaction, d, v, t_load=None, f_load=None, label='', color='black'):
    axs[0].set_ylabel('force')
    if t_load is not None:
        axs[0].plot(t_load, f_load, label='load', color='black')
    axs[0].plot(t, f_load, label='load', color=color)
    axs[0].legend()
    axs[1].set_ylabel('reaction')
    axs[1].plot(t, f_reaction,  label=label+' reaction', color=color)
    axs[1].legend()
    axs[2].set_ylabel('deformations')
    axs[2].plot(t, d, color=color)
    axs[3].set_ylabel('velocities')
    axs[3].plot(t, v, color=color)
    axs[3].set_xlabel('time')

