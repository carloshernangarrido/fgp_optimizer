import numpy as np

from models.chain_like import Model
from scipy.optimize import differential_evolution, Bounds, minimize


class Optimization:
    def __init__(self, base_model: Model, obj_fun: callable, lb: dict, ub: dict, uniform: bool = False):
        assert isinstance(base_model, Model) and callable(obj_fun) and isinstance(lb, dict) and isinstance(ub, dict)
        assert lb.keys() == ub.keys(), 'upper bound and lower bound must refer to the same parameters'
        self.model = base_model
        self.obj_func = obj_fun
        self.parameters = list(lb.keys())

        lb_values, ub_values = [], []
        for key in self.parameters:
            lb_values.append(lb[key])
            ub_values.append(ub[key])

        self.bounds = Bounds(lb=lb_values, ub=ub_values, keep_feasible=True)
        self.result = None

        self.initial_guess = []
        for i_x, element_name in enumerate(self.parameters):
            for element in self.model.mesh.elements:
                if element_name in element.aliases():
                    self.initial_guess.append(element.props['value'])
                    break
        self.initial_guess = np.array(self.initial_guess)
        self.model_list = []
        self.uniform = uniform
        self.element_names_uniform = {}
        for i_x, element_name in enumerate(self.parameters):
            if element_name.split('_')[0] not in \
                    [element_type.split('_')[0] for element_type in self.element_names_uniform]:
                self.element_names_uniform.update({element_name.split('_')[0]: i_x})

    def opt_obj_func(self, x: list = None):
        if x is not None:
            for i_x, element_name in enumerate(self.parameters):
                if self.uniform:
                    i_x = self.element_names_uniform[element_name.split('_')[0]]
                for element in self.model.mesh.elements:
                    if element_name in element.aliases():
                        element.props['value'] = x[i_x]
                        if element.element_type == 'muN':
                            element.update_props('c_th')
                        break
        self.model.update_model()
        self.model.solve()
        self.model_list.append(self.model.deepcopy())
        return self.obj_func(self.model)

    def optimize(self, maxiter=None, disp: bool = False, workers: int = 1, vectorized: bool = None, method: str = None):
        if method == 'differential_evolution':
            self.result = differential_evolution(self.opt_obj_func, bounds=self.bounds, disp=disp,
                                                 maxiter=maxiter, workers=workers, x0=self.initial_guess,
                                                 vectorized=vectorized)
        elif method == 'simplex':
            self.result = minimize(self.opt_obj_func, bounds=self.bounds, x0=self.initial_guess,
                                   options={'maxiter': maxiter, 'disp': disp}, method='Nelder-Mead')
        else:
            self.result = None
        return self.result

