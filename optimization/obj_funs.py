import numpy as np
from models import chain_like as cl
from input_parameters import fixed_dof, m, c, k, n_elements, max_mass_total

from input_parameters import lambda_, max_def_restriction


def obj_fun_peak(model_: cl.Model) -> float:
    return max(abs(model_.reactions(fixed_dof)))


def obj_fun_peak_and_deformation(model_: cl.Model) -> float:
    max_def = np.max(np.abs(model_.deformations()))
    if max_def < max_def_restriction:
        return obj_fun_peak(model_)
    else:
        return obj_fun_peak(model_) + lambda_*max_def


def obj_fun_impulse(model_: cl.Model) -> float:
    return abs(model_.impulses(fixed_dof)[-1])


def obj_fun_impulse_and_deformation(model_: cl.Model) -> float:
    max_def = np.max(np.abs(model_.deformations()))
    if max_def < max_def_restriction:
        return obj_fun_impulse(model_)
    else:
        return obj_fun_impulse(model_) + lambda_*max_def


def opt_obj_fun_override_density_uniform(x: np.ndarray, model: cl.Model):
    """
    :param x: densities and masses [dens, mass]
    :param model:
    :return: modified model_.
    """
    for element in model.mesh.elements:
        if element.__str__().split('_')[0] == 'c':
            element.props['value'] = x[0] * c
        if element.__str__().split('_')[0] == 'k':
            element.props['value'] = x[0] * k
        if element.__str__().split('_')[0] == 'm':
            element.props['value'] = x[1]
    return model


def opt_obj_fun_override_density_fg(x: np.ndarray, model: cl.Model):
    """
    :param x: densities and masses [dens_0_1, ..., dens_n-1_n, m_0_1, ..., m_n-1_n]
    :param model:
    :return: modified model_.
    """
    n_elements = model.n_dof - 1
    for element in model.mesh.elements:
        for i, c_i_j in enumerate([f"c_{i}_{i + 1}" for i in range(n_elements)]):
            if c_i_j in element.aliases():
                element.props['value'] = x[i] * c
        for i, k_i_j in enumerate([f"k_{i}_{i + 1}" for i in range(n_elements)]):
            if k_i_j in element.aliases():
                element.props['value'] = x[i] * k
        for i, m_i_j in enumerate([f"m_{i}_{i + 1}" for i in range(n_elements)]):
            if m_i_j in element.aliases():
                element.props['value'] = x[n_elements + i]
    return model


def restriction_fun_uniform(x: np.ndarray):
    return n_elements * x[1] - max_mass_total


def restriction_fun_fg(x: np.ndarray):
    return np.sum(x[n_elements:]) - max_mass_total

