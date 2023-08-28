import datetime
import logging
import pickle
import sys

from models import chain_like as cl
from optimization import optimizers as optim, bounds
from optimization.obj_funs import obj_fun_peak, opt_obj_fun_override_density_uniform, opt_obj_fun_override_density_fg, \
                                    restriction_fun_fg, restriction_fun_uniform
from plotting.results import plot_results


if __name__ == '__main__':
    from input_parameters import *
    logging.basicConfig(filename=f'log{opt_id}.txt', filemode='w', format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f"{peak_force=} N, {applied_impulse=} Ns")

    # Initial model
    mesh = (cl.Mesh(total_length, n_dof))
    mesh.fill_elements('k', k) \
        .fill_elements('penalty_gap', penalty_gap) \
        .fill_elements('c', c) \
        .fill_elements('m', max_mass_total/n_elements)
    mesh.elements[9].props['value'], mesh.elements[10].props['value'], mesh.elements[11].props['value'] = \
        min_mass_dof, min_mass_dof, max_mass_total - 2*min_mass_dof
    constraints = cl.Constraint(dof_s=fixed_dof)
    loads = cl.Load(dof_s=load_dof, t=t_vector, force=force_vector)
    initial_conditions = [cl.InitialCondition(dof0, 'displacement', d0),
                          cl.InitialCondition(dof0, 'velocity', v0)]
    op = {'t_vector': t_vector, 'method': 'BDF'}
    model = cl.Model(mesh=mesh, constraints=constraints, loads=loads, initial_conditions=initial_conditions, options=op)
    t_i = datetime.datetime.now()
    model.solve()
    logging.info(f"*** Initial model *** Elapsed time: {datetime.datetime.now() - t_i}")

    # BASE
    lb_values_uniform, ub_values_uniform = \
        bounds.bounds_values_density(n_elements=n_elements, min_mass=min_mass_dof, max_mass=max_mass_dof,
                                     min_rel_density=min_rel, max_rel_density=max_rel, uniform=True)
    opt_uniform = optim.ConstructiveOptimization(base_model=model.deepcopy(), obj_fun=obj_fun_peak,
                                                 lb_values=lb_values_uniform, ub_values=ub_values_uniform,
                                                 opt_obj_fun_override=opt_obj_fun_override_density_uniform,
                                                 restrictions_fun=restriction_fun_uniform)
    logging.info(f"\n*** base: {opt_uniform.opt_obj_func()}")
    logging.info(f"with: {[element.props['value'] for element in opt_uniform.model.mesh.elements]}")
    _, axs = plt.subplots(4, 1, sharex='all')
    plot_results(axs, t=model.sol.t, f_reaction=model.reactions(fixed_dof), impulse=model.impulses(fixed_dof),
                 d=model.displacements(dof0), v=model.velocities(dof0),
                 t_load=t_vector, f_load=force_vector, label='base')
    fig, ax, ani = model.animate()
    plt.show()

    # UNIFORM
    if flags['opt_uniform']:
        opt_uniform.optimize(maxiter=maxiter, disp=flags['disp'], method=flags['method'], workers=flags['workers'])
        with open(f'opt_uniform_{opt_id}.pk', 'wb') as file:
            pickle.dump(opt_uniform, file)
    else:
        with open(f'opt_uniform_{opt_id}.pk', 'rb') as file:
            opt_uniform = pickle.load(file)
    logging.info(f"\n*** uniform optimal: {opt_uniform.opt_obj_func()}")
    logging.info(f"with: {[element.props['value'] for element in opt_uniform.model.mesh.elements]}")
    _, axs = plt.subplots(4, 1, sharex='all')
    plot_results(axs, t=opt_uniform.model.sol.t,
                 f_reaction=opt_uniform.model.reactions(fixed_dof), impulse=opt_uniform.model.impulses(fixed_dof),
                 d=opt_uniform.model.displacements(dof0), v=opt_uniform.model.velocities(dof0),
                 t_load=t_vector, f_load=force_vector, label='base')
    fig, ax, ani = opt_uniform.model.animate()
    plt.show()

    # FUNCTIONALLY GRADED
    if flags['opt_fg']:
        lb_values_fg, ub_values_fg = \
            bounds.bounds_values_density(n_elements=n_elements, min_mass=min_mass_dof, max_mass=max_mass_dof,
                                         min_rel_density=min_rel, max_rel_density=max_rel, uniform=False)
        opt_fg = optim.ConstructiveOptimization(base_model=model.deepcopy(), obj_fun=obj_fun_peak,
                                                lb_values=lb_values_fg, ub_values=ub_values_fg,
                                                opt_obj_fun_override=opt_obj_fun_override_density_fg,
                                                initial_guess=(n_dof-1)*[opt_uniform.result.x[0]] +
                                                              (n_dof-1)*[opt_uniform.result.x[1]],
                                                restrictions_fun=restriction_fun_fg)
        opt_fg.optimize(maxiter=maxiter, disp=flags['disp'], method=flags['method'], workers=flags['workers'])
        with open(f'opt_fg_{opt_id}.pk', 'wb') as file:
            pickle.dump(opt_fg, file)
    else:
        with open(f'opt_fg_{opt_id}.pk', 'rb') as file:
            opt_fg = pickle.load(file)
    logging.info(f"\n*** functionally graded optimal: {opt_fg.opt_obj_func()}")
    logging.info(f"with, {[element.props['value'] for element in opt_fg.model.mesh.elements]}")

    _, axs = plt.subplots(4, 1, sharex='all')
    plot_results(axs, t=model.sol.t, f_reaction=model.reactions(fixed_dof), impulse=model.impulses(fixed_dof),
                 d=model.displacements(dof0), v=model.velocities(dof0),
                 label='base')
    plot_results(axs, t=opt_uniform.model.sol.t,
                 f_reaction=opt_uniform.model.reactions(fixed_dof), impulse=opt_uniform.model.impulses(fixed_dof),
                 d=opt_uniform.model.displacements(dof0), v=opt_uniform.model.velocities(dof0),
                 label='opt. uniform')
    plot_results(axs, t=opt_fg.model.sol.t,
                 f_reaction=opt_fg.model.reactions(fixed_dof), impulse=opt_fg.model.impulses(fixed_dof),
                 d=opt_fg.model.displacements(dof0), v=opt_fg.model.velocities(dof0),
                 t_load=t_vector, f_load=force_vector, label='opt FG')
    fig, ax, ani = opt_fg.model.animate()
    plt.show()

