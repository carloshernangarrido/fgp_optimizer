from typing import Union, List
import numpy as np


element_types: dict = {'m': {'description': 'mass',
                             'props': ['value']},
                       'k': {'description': 'linear stiffness',
                             'props': ['value']},
                       'c': {'description': 'linear viscous damping',
                             'props': ['value']},
                       'muN': {'description': 'Coulomb friction',
                               'props': ['value']},
                       'gap': {'description': 'closing gap contact',
                               'props': ['value', 'contact_stiffness']}}

constraint_types = ['imposed_displacement']

initial_condition_types = ['displacement', 'velocity']


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

    def force_ij(self, u, u_dot):
        """
        Returns the force exerted by the element depending on the displacement and velocities, and retain them in the
        attribute response_.

        :param u: vector of displacements.
        :param u_dot: vector of velocities.
        :return: float force exerted by the element.
        """
        if self.element_type == 'm':
            raise ValueError('inertial pseudo-force is not supported')
        try:
            if self.element_type == 'k':
                return self.props['value']*(u[self.i] - u[self.j])
            elif self.element_type == 'c':
                return self.props['value']*(u_dot[self.i] - u_dot[self.j])
            elif self.element_type == 'muN':
                return self.props['value']*np.sign(u_dot[self.i] - u_dot[self.j])
            elif self.element_type == 'gap':
                contact_deformation = (u[self.i] - u[self.j]) - self.props['value']
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
        self.coordinates = np.linspace(0, length, n_dof)
        self.displacements = np.zeros(n_dof)
        self.velocities = np.zeros(n_dof)
        element_mass = total_mass/(n_dof - 1)
        self.elements = [Element('m', i=i, j=i+1, props=element_mass) for i in range(self.n_dof - 1)]

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


class DofWise:
    def __init__(self, dof_s: Union[int, List[int]]):
        if isinstance(dof_s, int):
            dof_s = [dof_s]
        assert all([isinstance(dof, int) and dof >= 0 for dof in dof_s])
        self.dof_s = dof_s


class Constraint(DofWise):
    def __init__(self, dof_s, constraint_type: str = 'imposed_displacement', value: Union[int, float] = 0.0):
        super().__init__(dof_s)
        assert constraint_type in constraint_types
        assert isinstance(value, (float, int))

        self.constraint_type = constraint_type
        self.value = float(value)


class Load(DofWise):
    def __init__(self, dof_s, t, force):
        super().__init__(dof_s)
        assert np.ndim(t) == 1, 't must be 1D array like'
        assert np.ndim(force) == 1, 'force must be 1D array like'
        assert len(t) == len(force)

        self.t = np.array(t)
        self.force = np.array(force)


class InitialCondition(DofWise):
    """
    All the initial conditions (displacements and velocities) are assumed 0.0, except for those defined using this.
    """
    def __init__(self, dof_s, ic_type: str = 'displacement', value: Union[float, int] = 0.0):
        super().__init__(dof_s)
        assert ic_type in initial_condition_types, f"Supported initial conditions types are: {initial_condition_types}, " \
                                                   f"but {ic_type} was specified."
        self.ic_type = ic_type
        self.value = float(value)


class Model:
    def __init__(self, mesh: Mesh, constraints: Union[Constraint, List[Constraint]] = None,
                 loads: Union[Load, List[Load]] = None):
        assert isinstance(mesh, Mesh)
        assert isinstance(constraints, (list, Constraint)) or constraints is None
        assert isinstance(loads, (list, Load)) or loads is None

        self.mesh = mesh
        self.constraints = [] if constraints is None else constraints
        self.constraints = [self.constraints] if isinstance(constraints, Constraint) else constraints
        self.loads = [] if loads is None else loads
        self.loads = [self.loads] if isinstance(loads, Load) else self.loads

        assert all([isinstance(constraint, Constraint) for constraint in self.constraints])
        assert all([isinstance(load, Load) for load in self.loads])

        self.n_dof = self.mesh.n_dof
        self.n_elements = len(self.mesh.elements)
        mass_elements = \
            [self.mesh.elements[i] for i in range(self.n_elements) if self.mesh.elements[i].element_type == 'm']
        self.dof_masses = np.zeros(self.n_dof)
        for mass_element in mass_elements:
            self.dof_masses[mass_element.i] += mass_element.props['value']
            self.dof_masses[mass_element.j] += mass_element.props['value']
        # massless elements connected to each dof
        self.connectivity = [[i_e for i_e in range(self.n_elements) if
                              (self.mesh.elements[i_e].element_type != 'm') and
                              (self.mesh.elements[i_e].i == i_dof or self.mesh.elements[i_e].j == i_dof)]
                             for i_dof in range(self.n_dof)]

    def f(self, t: float, y: np.ndarray):
        """
        Function characterizing the set of ordinary differential equations defined as:

        dy / dt = f(t, y)
        y(t0) = y0

        :param t: time
        :param y: 1D vector of system states. y = [ u, u_dot ]
        :returns y_dot: 1D vector of derivative system states. y = [ u_dot, -forces_sum/masses ]
        """
        forces_sum = np.zeros(self.n_dof)
        # element forces
        for i_dof in range(self.n_dof):
            for i_e in self.connectivity[i_dof]:
                forces_sum[i_dof] += self.mesh.elements[i_e].force_ij(u=y[0:self.n_dof],
                                                                      u_dot=y[self.n_dof:])
        # y_dot = np.hstack((y[0:self.n_dof], ...))
        # return y_dot
