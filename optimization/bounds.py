import numpy as np


def bounds_values_density(n_elements, min_mass: float, max_mass: float,
                          min_rel_density: float = 0.01, max_rel_density: float = 1.00,
                          uniform: bool = False):
    if uniform:  # densities and masses [dens, mass]
        lb_values = np.array([min_rel_density, min_mass])
        ub_values = np.array([max_rel_density, max_mass])
    else:  # densities and masses [dens_0_1, ..., dens_n-1_n, m_0_1, ..., m_n-1_n]
        lb_values = np.hstack((n_elements*[min_rel_density], n_elements*[min_mass]))
        ub_values = np.hstack((n_elements*[max_rel_density], n_elements*[max_mass]))
    return lb_values, ub_values


def bounds(min_rel: float = .01, max_rel: float = 1,
           c: float = None, m: float = None, muN: dict = None):
    if c is not None and m is not None:
        lb = {'c_0_1': min_rel * c,
              'c_1_2': min_rel * c,
              'c_2_3': min_rel * c,
              'c_3_4': min_rel * c,
              'm_0_1': min_rel * m,
              'm_1_2': min_rel * m,
              'm_2_3': min_rel * m,
              'm_3_4': min_rel * m}
        ub = {'c_0_1': max_rel * c,
              'c_1_2': max_rel * c,
              'c_2_3': max_rel * c,
              'c_3_4': max_rel * c,
              'm_0_1': max_rel * m,
              'm_1_2': max_rel * m,
              'm_2_3': max_rel * m,
              'm_3_4': max_rel * m}
    elif muN is not None:
        lb = {'muN_0_1': min_rel * muN['value'],
              'muN_1_2': min_rel * muN['value'],
              'muN_2_3': min_rel * muN['value'],
              'muN_3_4': min_rel * muN['value']}
        ub = {'muN_0_1': max_rel * muN['value'],
              'muN_1_2': max_rel * muN['value'],
              'muN_2_3': max_rel * muN['value'],
              'muN_3_4': max_rel * muN['value']}
    elif m is not None:
        lb = {'m_0_1': min_rel * m,
              'm_1_2': min_rel * m,
              'm_2_3': min_rel * m,
              'm_3_4': min_rel * m}
        ub = {'m_0_1': max_rel * m,
              'm_1_2': max_rel * m,
              'm_2_3': max_rel * m,
              'm_3_4': max_rel * m}
    else:
        raise NotImplementedError
    return lb, ub
