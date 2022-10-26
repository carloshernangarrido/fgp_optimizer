# Viscoelastic foam
# Taken from: Ge, C., & Rice, B. (2018). Impact damping ratio of a nonlinear viscoelastic foam. Polymer Testing,
# 72(August), 187–195. https://doi.org/10.1016/j.polymertesting.2018.10.023
import numpy as np

# Young modulus a low strain rate
E_prime = 200e3  # (Pa) Section 6. Conclusions

# Loss tangent = 0.17 at 1 Hz Section 6. Conclusions
tan_delta_1Hz = 0.17  # E = E_prime + i*E_second = E_prime (1 + i*tan_delta)


# # Element dimensions
# a = 0.10*0.10  # (m2) cross-section area
# l = 0.01  # (m) length

def kc_from_al(area: float, length: float):
    # Stiffness
    k = E_prime * area / length

    # Damping coefficient
    E_second = E_prime * tan_delta_1Hz
    pi = np.pi
    f = 1
    c = (E_second * area / length) / (2 * pi * f)
    return k, c
