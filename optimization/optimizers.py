import numpy as np

from models.chain_like import Model
from scipy.optimize import differential_evolution, Bounds, minimize


class BaseOptimization:
    def __init__(self, base_model: Model, obj_fun: callable):
        assert isinstance(base_model, Model) and callable(obj_fun)
        self.model = base_model
        self.obj_func = obj_fun
        self.model_list = []
        self.result = None

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


class Optimization(BaseOptimization):
    def __init__(self, base_model: Model, obj_fun: callable, lb: dict, ub: dict, uniform: bool = False,
                 opt_obj_fun_override: callable = None):
        """
        Class for optimization of models.

        :param base_model:
        :param obj_fun:
        :param lb:
        :param ub:
        :param uniform:
        :param opt_obj_fun_override: a callable that accepts x (the optimization vector) and a model, and returns the
        model with modified parameters. It must have the following signature:

        >>> def opt_obj_fun_override_(x: np.ndarray, model: Model):
        >>>     ...
        >>>     return model
        """
        super().__init__(base_model, obj_fun)

        assert isinstance(lb, dict) and isinstance(ub, dict)
        assert lb.keys() == ub.keys(), 'upper bound and lower bound must refer to the same parameters'
        self.parameters = list(lb.keys())

        lb_values, ub_values = [], []
        for key in self.parameters:
            lb_values.append(lb[key])
            ub_values.append(ub[key])

        self.bounds = Bounds(lb=lb_values, ub=ub_values, keep_feasible=True)

        self.initial_guess = []
        for i_x, element_name in enumerate(self.parameters):
            for element in self.model.mesh.elements:
                if element_name in element.aliases():
                    self.initial_guess.append(element.props['value'])
                    break
        self.initial_guess = np.array(self.initial_guess)

        self.uniform = uniform
        self.element_names_uniform = {}
        for i_x, element_name in enumerate(self.parameters):
            if element_name.split('_')[0] not in \
                    [element_type.split('_')[0] for element_type in self.element_names_uniform]:
                self.element_names_uniform.update({element_name.split('_')[0]: i_x})

        assert callable(opt_obj_fun_override) or opt_obj_fun_override is None, \
            "opt_obj_fun_override must be callable or None"
        self.opt_obj_fun_override = opt_obj_fun_override

    def opt_obj_func(self, x: list = None):
        if x is None:  # just wondering the output for current model
            self.model.update_model()
            self.model.solve()
            self.model_list.append(self.model.deepcopy())
            return self.obj_func(self.model)
        if self.opt_obj_fun_override is None:
            for i_x, element_name in enumerate(self.parameters):
                if self.uniform:
                    i_x = self.element_names_uniform[element_name.split('_')[0]]
                for element in self.model.mesh.elements:
                    if element_name in element.aliases():
                        element.props['value'] = x[i_x]
                        if element.element_type == 'muN':
                            element.update_props('c_th')
                        elif element.element_type == 'penalty_gap':
                            element.update_props('quadratic_coefficient')
                        break
            self.model.update_model()
            self.model.solve()
            self.model_list.append(self.model.deepcopy())
            return self.obj_func(self.model)
        else:
            ret = self.opt_obj_fun_override(x, self.model)
            if isinstance(ret, Model):
                self.model = ret
                self.model.update_model()
                self.model.solve()
                self.model_list.append(self.model.deepcopy())
                return self.obj_func(self.model)
            else:  # restriction was violated
                return np.inf


class ConstructiveOptimization(BaseOptimization):
    def __init__(self, base_model: Model, obj_fun: callable, bounds: Bounds, opt_obj_fun_override: callable = None):
        super().__init__(base_model, obj_fun)
        ...
