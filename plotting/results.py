import numpy as np
import scipy as sp

from models.chain_like import Model


def plot_results(axs, t, f_reaction, d, v, t_load=None, f_load=None, label='', color='black'):
    axs[0].set_ylabel('force')
    if t_load is not None:
        axs[0].plot(t_load, f_load, color='black')
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


def plot_fg(axs, model: Model, label: str = ""):
    k_pos, c_pos, m_pos, k_val, c_val, m_val = [], [], [], [], [], []
    for element in model.mesh.elements:
        if element.element_type == 'k':
            k_pos.append(model.mesh.coordinates[element.i])
            k_val.append(element.props['value'])
            k_pos.append(model.mesh.coordinates[element.j])
            k_val.append(element.props['value'])
        elif element.element_type == 'c':
            c_pos.append(model.mesh.coordinates[element.i])
            c_val.append(element.props['value'])
            c_pos.append(model.mesh.coordinates[element.j])
            c_val.append(element.props['value'])
    for m_pos_, m_val_ in enumerate(model.dof_masses):
        m_pos.append(model.mesh.coordinates[m_pos_])
        m_val.append(m_val_)
    axs[0].set_ylabel('stiffness (N/m)')
    axs[0].plot(k_pos, k_val, label=label)
    axs[1].set_ylabel('damping coefficient (Ns/m)')
    axs[1].plot(c_pos, c_val)
    axs[2].set_ylabel('mass (kg)')
    axs[2].plot(m_pos, m_val, linestyle='', marker='x')
    [axs[i].grid() for i in range(3)]
    axs[0].legend()

