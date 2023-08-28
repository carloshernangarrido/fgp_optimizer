# Viscoelastic foam
import numpy as np


def kcgapkc_from_al(area: float, length: float, material: str):
    """
    stiffness, damping coefficient and gap
    :param area:
    :param length:
    :param material:
    :return: k (stiffness), c (damping coefficient), and gap (gap)
    """
    k, c, gap, kc = None, None, 1.0*length, np.inf
    if material == 'literature_viscoelastic_foam':
        # Taken from: Ge, C., & Rice, B. (2018). Impact damping ratio of a nonlinear viscoelastic foam. Polymer Testing,
        # 72(August), 187–195. https://doi.org/10.1016/j.polymertesting.2018.10.023
        # Young modulus a low strain rate
        E_prime = 200e3  # (Pa) Section 6. Conclusions
        # Loss tangent = 0.17 at 1 Hz Section 6. Conclusions
        tan_delta_1Hz = 0.17  # E = E_prime + i*E_second = E_prime (1 + i*tan_delta)
        # Stiffness
        k = E_prime * area / length
        # Damping coefficient
        E_second = E_prime * tan_delta_1Hz
        pi = np.pi
        f = 1
        c = (E_second * area / length) / (2 * pi * f)
    elif material == 'characterized_viscoelastic_foam':
        k_m3_m3_m024 = 5e4  # N/m (post compaction 2e6, 40 times)
        c_m3_m3_m024 = 1e3  # Ns/m
        E_prime = k_m3_m3_m024 * .048 / (.3*.3)
        # Stiffness
        k = E_prime * area / length
        # Damping coefficient
        E_damping = c_m3_m3_m024 * .048 / (.3*.3)
        c = E_damping * area / length
        gap = (.030/.048)*length
        kc = 40*k
    return k, c, gap, kc


def m_from_al(area: float, length: float, material: str):
    if material == 'lead':
        return 11000.0 * area * length
    elif material == 'steel':
        return 7850 * area * length
