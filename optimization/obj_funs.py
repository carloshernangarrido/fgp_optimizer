import numpy as np

from models import chain_like as cl


def obj_fun_peak(model_: cl.Model):
    return max(abs(model_.reactions(fixed_dof)))


def obj_fun_impulse(model_: cl.Model):
    return max(abs(model_.impulses(fixed_dof)))


def opt_obj_fun_override(x: np.ndarray, model: cl.Model):
    """
    :param x: damping coefficients and masses [c_0_1, ..., c_n-1_n, m_0_1, ..., m_n-1_n]
    :param model:
    :return: modified model_ of np.inf if some restriction is violated
    """
    n_elements = len(x) // 2
    max_mass = m * n_elements
    if sum(x[n_elements:]) > max_mass:
        return np.inf
    else:
        for element in model.mesh.elements:
            for i, c_i_j in enumerate([f"c_{i}_{i + 1}" for i in range(n_elements)]):
                if c_i_j in element.aliases():
                    element.props['value'] = x[i]
            for i, m_i_j in enumerate([f"m_{i}_{i + 1}" for i in range(n_elements)]):
                if m_i_j in element.aliases():
                    element.props['value'] = x[n_elements + i]
    return model


def opt_obj_fun_override_density(x: np.ndarray, model: cl.Model):
    """
    :param x: densities amd masses [dens_0_1, ..., dens_n-1_n, m_0_1, ..., m_n-1_n]
    :param model:
    :return: modified model_ of np.inf if some restriction is violated
    """
    n_elements = len(x) // 2
    max_mass = m * n_elements
    if sum(x[n_elements:]) > max_mass:
        return np.inf
    else:
        for element in model.mesh.elements:
            for i, c_i_j in enumerate([f"c_{i}_{i + 1}" for i in range(n_elements)]):
                if c_i_j in element.aliases():
                    element.props['value'] = x[i]
            for i, m_i_j in enumerate([f"m_{i}_{i + 1}" for i in range(n_elements)]):
                if m_i_j in element.aliases():
                    element.props['value'] = x[n_elements + i]
    return model


from main import fixed_dof, m
