import datetime
import logging
import pickle
import sys

from models import chain_like as cl
from optimization import optimizers as optim
from plotting.results import plot_results

if __name__ == '__main__':
    from input_parameters import *

    if flags['obj_fun'] == 'peak':
        from optimization.obj_funs import obj_fun_peak_and_deformation as obj_fun
    elif flags['obj_fun'] == 'impulse':
        from optimization.obj_funs import obj_fun_impulse_and_deformation as obj_fun
    if flags['fun_override'] == 'density':
        from optimization.obj_funs import opt_obj_fun_override_density_uniform as opt_obj_fun_override_uniform
        from optimization.obj_funs import opt_obj_fun_override_density_fg as opt_obj_fun_override_fg
        from optimization.bounds import bounds_values_density as bounds_values
        from optimization.obj_funs import restriction_fun_density_fg as restriction_fun_fg
        from optimization.obj_funs import restriction_fun_density_uniform as restriction_fun_uniform
    elif flags['fun_override'] == 'denskc_m':
        from optimization.obj_funs import opt_obj_fun_override_denskc_m_uniform as opt_obj_fun_override_uniform
        from optimization.obj_funs import opt_obj_fun_override_denskc_m_fg as opt_obj_fun_override_fg
        from optimization.bounds import bounds_values_denskc_m as bounds_values
        from optimization.obj_funs import restriction_fun_denskc_m_fg as restriction_fun_fg
        from optimization.obj_funs import restriction_fun_denskc_m_uniform as restriction_fun_uniform

    logging.basicConfig(filename=f'log{opt_id}.txt', filemode='w', format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    with open('input_parameters.py', 'r') as input_file_handler:
        logging.info(input_file_handler.read())
    logging.info(f"{peak_force=} N, {applied_impulse=} Ns")

    # Initial model
    mesh = (cl.Mesh(total_length, n_dof))
    mesh.fill_elements('k', k) \
        .fill_elements('penalty_gap', penalty_gap) \
        .fill_elements('c', c) \
        .fill_elements('m', min_mass_dof)
    constraints = cl.Constraint(dof_s=fixed_dof)
    loads = cl.Load(dof_s=load_dof, t=t_vector, force=force_vector)
    initial_conditions = [cl.InitialCondition(dof0, 'displacement', d0),
                          cl.InitialCondition(dof0, 'velocity', v0)]
    try:
        logging.info(f'protected_structure is defined as {protected_structure}')
        for element in mesh.elements:
            if element.element_type == 'm' and element.i == 0 and element.j == 1:
                element.props['value'] = protected_structure['m']*2  # half of this mass is lost in the fix support
            elif element.element_type == 'k' and element.i == 0 and element.j == 1:
                element.props['value'] = protected_structure['k']
            elif element.element_type == 'c' and element.i == 0 and element.j == 1:
                element.props['value'] = protected_structure['c']
        if flags['fun_override'] == 'density':
            from optimization.obj_funs import opt_obj_fun_override_density_uniform_protstr as opt_obj_fun_override_uniform
            from optimization.obj_funs import opt_obj_fun_override_density_fg_protstr as opt_obj_fun_override_fg
            from optimization.bounds import bounds_values_density as bounds_values
            from optimization.obj_funs import restriction_fun_density_fg_protstr as restriction_fun_fg
            from optimization.obj_funs import restriction_fun_density_uniform_protstr as restriction_fun_uniform
        elif flags['fun_override'] == 'denskc_m':
            from optimization.obj_funs import opt_obj_fun_override_denskc_m_uniform_protstr as opt_obj_fun_override_uniform
            from optimization.obj_funs import opt_obj_fun_override_denskc_m_fg_protstr as opt_obj_fun_override_fg
            from optimization.bounds import bounds_values_denskc_m as bounds_values
            from optimization.obj_funs import restriction_fun_denskc_m_fg_protstr as restriction_fun_fg
            from optimization.obj_funs import restriction_fun_denskc_m_uniform_protstr as restriction_fun_uniform
    except NameError:
        logging.info('protected_structure is not defined')

    op = {'t_vector': t_vector, 'method': 'BDF'}
    model = cl.Model(mesh=mesh, constraints=constraints, loads=loads, initial_conditions=initial_conditions, options=op)
    t_i = datetime.datetime.now()
    model.solve()
    logging.info(f"*** Initial model *** Elapsed time: {datetime.datetime.now() - t_i}")

    # BASE
    lb_values_uniform, ub_values_uniform = \
        bounds_values(n_elements=n_elements, min_mass=min_mass_dof, max_mass=max_mass_dof,
                      min_rel_density=min_rel, max_rel_density=max_rel, uniform=True)
    opt_uniform = optim.ConstructiveOptimization(base_model=model.deepcopy(), obj_fun=obj_fun,
                                                 lb_values=lb_values_uniform, ub_values=ub_values_uniform,
                                                 opt_obj_fun_override=opt_obj_fun_override_uniform,
                                                 restrictions_fun=restriction_fun_uniform)
    logging.info(f"*** base: {opt_uniform.opt_obj_func()}")
    logging.info(f"with: {[element.props['value'] for element in opt_uniform.model.mesh.elements]}")
    _, axs = plt.subplots(4, 1, sharex='all')
    plot_results(axs, t=model.sol.t, f_reaction=model.reactions(fixed_dof), impulse=model.impulses(fixed_dof),
                 d=model.deformations(), v=model.velocities(dof0),
                 t_load=t_vector, f_load=force_vector, label='base', color='blue')
    fig, ax, ani = model.animate(each=animate_each)
    plt.show()

    # UNIFORM
    if flags['opt_uniform']:
        opt_uniform.optimize(maxiter=maxiter, disp=flags['disp'], method=flags['method'], workers=flags['workers'])
        with open(f'opt_uniform_{opt_id}.pk', 'wb') as file:
            pickle.dump(opt_uniform, file)
    else:
        with open(f'opt_uniform_{opt_id}.pk', 'rb') as file:
            opt_uniform = pickle.load(file)
    logging.info(f"*** uniform optimal: {opt_uniform.opt_obj_func()}")
    logging.info(f"with: {[element.props['value'] for element in opt_uniform.model.mesh.elements]}")
    _, axs = plt.subplots(4, 1, sharex='all')
    plot_results(axs, t=opt_uniform.model.sol.t,
                 f_reaction=opt_uniform.model.reactions(fixed_dof), impulse=opt_uniform.model.impulses(fixed_dof),
                 d=opt_uniform.model.deformations(), v=opt_uniform.model.velocities(dof0),
                 t_load=t_vector, f_load=force_vector, label='base', color='blue')
    fig, ax, ani = opt_uniform.model.animate(each=animate_each)
    plt.show()

    # FUNCTIONALLY GRADED
    if flags['opt_fg']:
        try:
            logging.info(f'protected_structure is defined as {protected_structure}')
            lb_values_fg, ub_values_fg = \
                bounds_values(n_elements=n_elements-1, min_mass=min_mass_dof, max_mass=max_mass_dof,
                              min_rel_density=min_rel, max_rel_density=max_rel, uniform=False)
            initial_guess = np.array([(n_dof - 2) * [_] for _ in opt_uniform.result.x]).flatten()
        except NameError:
            lb_values_fg, ub_values_fg = \
                bounds_values(n_elements=n_elements, min_mass=min_mass_dof, max_mass=max_mass_dof,
                              min_rel_density=min_rel, max_rel_density=max_rel, uniform=False)
            initial_guess = np.array([(n_dof - 1) * [_] for _ in opt_uniform.result.x]).flatten()
        opt_fg = optim.ConstructiveOptimization(base_model=model.deepcopy(), obj_fun=obj_fun,
                                                lb_values=lb_values_fg, ub_values=ub_values_fg,
                                                opt_obj_fun_override=opt_obj_fun_override_fg,
                                                initial_guess=initial_guess,
                                                restrictions_fun=restriction_fun_fg)
        opt_fg.optimize(maxiter=maxiter, disp=flags['disp'], method=flags['method'], workers=flags['workers'])
        with open(f'opt_fg_{opt_id}.pk', 'wb') as file:
            pickle.dump(opt_fg, file)
    else:
        with open(f'opt_fg_{opt_id}.pk', 'rb') as file:
            opt_fg = pickle.load(file)
    logging.info(f"*** functionally graded optimal: {opt_fg.opt_obj_func()}")
    logging.info(f"with, {[element.props['value'] for element in opt_fg.model.mesh.elements]}")

    _, axs = plt.subplots(4, 1, sharex='all')
    plot_results(axs, t=model.sol.t, f_reaction=model.reactions(fixed_dof), impulse=model.impulses(fixed_dof),
                 d=model.deformations(), v=model.velocities(dof0),
                 label='base', color='blue')
    plot_results(axs, t=opt_uniform.model.sol.t,
                 f_reaction=opt_uniform.model.reactions(fixed_dof), impulse=opt_uniform.model.impulses(fixed_dof),
                 d=opt_uniform.model.deformations(), v=opt_uniform.model.velocities(dof0),
                 label='opt. uniform', color='red')
    plot_results(axs, t=opt_fg.model.sol.t,
                 f_reaction=opt_fg.model.reactions(fixed_dof), impulse=opt_fg.model.impulses(fixed_dof),
                 d=opt_fg.model.deformations(), v=opt_fg.model.velocities(dof0),
                 t_load=t_vector, f_load=force_vector, label='opt FG', color='green')
    fig, ax, ani = opt_fg.model.animate(each=animate_each)
    plt.show()
    ...
