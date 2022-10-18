from typing import Union
import numpy as np


element_types: dict = {'k': 'linear stiffness',
                       'c': 'linear viscous damping',
                       'muN': 'Coulomb friction',
                       'gap': 'closing gap contact'}


class Element:
    def __init__(self, element_type: str, i: int, j: int, props: Union[None, dict]):
        if props is None:
            props = {'value': 0.0}
        assert element_type in element_types.keys(), 'Unsupported element_type'
        self.element_type = element_type
        self.i = i
        self.j = j
        self.props = props

    def __str__(self, ji: bool = False):
        return f'{self.element_type}_{self.j}_{self.i}' if ji else f'{self.element_type}_{self.i}_{self.j}'

    def force_ij(self, **kwargs):
        """
        Force exerted by the element depending on the displacement and velocities.

        :param kwargs: dict containing 2 or more responses from: 'u_i', 'u_j', 'u_dot_i', 'u_dot_j', as float.
        :return: float force exerted by the element.
        """
        try:
            if self.element_type == 'k':
                return self.props['value']*(kwargs['u_i'] - kwargs['u_j'])
            elif self.element_type == 'c':
                return self.props['value']*(kwargs['u_dot_i'] - kwargs['u_dot_j'])
            elif self.element_type == 'muN':
                return self.props['value']*np.sign(kwargs['u_dot_i'] - kwargs['u_dot_j'])
            elif self.element_type == 'gap':
                contact_deformation = (kwargs['u_i'] - kwargs['u_j']) - self.props['value']
                if contact_deformation > 0:
                    return self.props['contact_stiffness']*contact_deformation
                else:
                    return 0.0
        except KeyError:
            raise KeyError('Insufficient responses to calculate element force.')

    def aliases(self):
        return [self.__str__(), self.__str__(ji=True)]

    def is_same_as(self, compare_element):
        return True if compare_element.__str__() in self.aliases() else False


class Mesh:
    def __init__(self, length: float, n_dof: int, total_mass: float):
        """
        Creates a Mesh object that represents a 1D deformable body
        :param total_mass: Total mass of the body
        :param length: Total length of the body
        :param n_dof: Number of degrees of freedom.
        """
        self.n_dof = n_dof
        self.masses = np.array(n_dof*[total_mass/n_dof])
        self.coordinates = np.linspace(0, length, n_dof)
