from typing import Union, Tuple, List, Dict
import numpy as np


element_types: dict = {'k': {'description': 'linear stiffness',
                             'props': ['value']},
                       'c': {'description': 'linear viscous damping',
                             'props': ['value']},
                       'muN': {'description': 'Coulomb friction',
                               'props': ['value']},
                       'gap': {'description': 'closing gap contact',
                               'props': ['value', 'contact_stiffness']}}


class Element:
    def __init__(self, element_type: str, i: int, j: int, props: Union[None, dict, float, int]):
        assert element_type in element_types.keys(), 'Unsupported element_type'
        if props is None:
            props = {'value': 0.0}
        elif isinstance(props, (float, int)):
            props = {'value': float(props)}
        elif isinstance(props, dict):
            assert all([req_prop in props.keys() for req_prop in element_types[element_type]['props']]), 'Required prop'
            assert all([prop in element_types[element_type]['props'] for prop in props.keys()]), 'Unsupported prop'
            assert all([isinstance(props[given_prop], (float, int)) for given_prop in props.keys()]), 'props must be ' \
                                                                                                      'numeric '
        else:
            raise AssertionError('props must be None, dict or numeric')
        self.element_type = element_type
        self.i = i
        self.j = j
        self.props = props
        self.response_ = None

    def __str__(self, ji: bool = False):
        return f'{self.element_type}_{self.j}_{self.i}' if ji else f'{self.element_type}_{self.i}_{self.j}'

    def force_ij(self, **kwargs):
        """
        Returns the force exerted by the element depending on the displacement and velocities, and retain them in the
        attribute response_.

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
            self.response_ = kwargs
        except KeyError:
            raise KeyError('Insufficient responses to calculate element force.')

    def aliases(self):
        return [self.__str__(), self.__str__(ji=True)]

    def is_same_as(self, compare_element):
        return True if compare_element.__str__() in self.aliases() else False


class Mesh:
    def __init__(self, length: Union[float, int], n_dof: int, total_mass: Union[float, int]):
        """
        Creates a Mesh object that represents a 1D deformable body.

        :param total_mass: Total mass of the body
        :param length: Total length of the body
        :param n_dof: Number of degrees of freedom.
        """
        assert isinstance(length, (int, float)) and length > 0
        assert isinstance(n_dof, int) and n_dof > 0
        assert isinstance(total_mass, (int, float)) and total_mass > 0

        self.n_dof = n_dof
        self.masses = np.array(n_dof*[total_mass/n_dof])
        self.coordinates = np.linspace(0, length, n_dof)
        self.displacements = np.zeros(n_dof)
        self.velocities = np.zeros(n_dof)
        self.elements = []

    def fill_elements(self, element_type, props_s: Union[List[Union[dict, float, int]], dict, float, int]):
        """
        Fill with elements all the mesh, assigning the values provided in values.

        :param element_type: Type of elements
        :param props_s: If it is a list, it has to be the same length as the n_dof-1.
        If it is a dict of numbers, these same numbers are used as props for all the elements. If it is numeric, this
        number is used as 'value' in props of all the elements.
        :return: The updated mesh
        """

        if isinstance(props_s, list) and len(props_s) == self.n_dof - 1:
            for i, props in enumerate(props_s):
                self.elements.append(Element(element_type, i=i, j=i+1, props=props))
        elif isinstance(props_s, (float, int, dict)):
            for i in range(self.n_dof - 1):
                self.elements.append((Element(element_type, i=i, j=i+1, props=props_s)))
        else:
            raise TypeError
        return self


class Constraint:
    def __init__(self, dofs: List[int], constraint_type: str, value: Union[int, float] = 0.0):
        self.dofs = dofs
        self.constraint_type = constraint_type
        self.value = float(value)


class Load:
    def __init__(self):
        ...


class Model:
    def __init__(self, mesh: Mesh, constraints: List[Constraint] = None, loads: List[Load] = None):
        assert isinstance(mesh, Mesh)
        assert isinstance(constraints, (list, Constraint)) or constraints is None
        assert isinstance(loads, (list, Load)) or loads is None

        self.mesh = mesh
        self.constraints = [] if constraints is None else constraints
        self.loads = [] if loads is None else loads

        assert all([isinstance(constraint, Constraint) for constraint in self.constraints])
        assert all([isinstance(load, Load) for load in self.loads])
