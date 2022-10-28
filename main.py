import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt

from material_properties import kc_from_al, m_from_al
from models import chain_like as cl
from optimization import optimizers as optim, bounds
from optimization.obj_funs import obj_fun_peak, opt_obj_fun_override
from plotting.results import plot_results

# Time
t_ini = 0.0
t_fin = .05
delta_t = .0001
t_vector = np.arange(t_ini, t_fin, delta_t)
t_N = len(t_vector)

area = 0.1 * 0.1
n_dof = 5
total_length = 0.10
element_length = total_length / (n_dof - 1)

min_rel, max_rel = 0.01, 1.00
k, c = kc_from_al(area=area, length=element_length, material='viscoelastic_foam')
m = m_from_al(area=area, length=0.001, material='lead')

# Plastification
# muN = .06
displacement_tol = .01
# muN = {'value': muN,
#        'v_th': element_length*displacement_tol/(t_fin-t_ini)}
# Compaction
# gap = {'value': 0.75 * element_length,
#        'contact_stiffness': 10000 * k}
penalty_gap = {'value': 0.75 * element_length,
               'contact_stiffness': 10000 * k,
               'penetration': 0.01 * 0.75 * element_length}

# Load: triangular pulse
peak_force = 200e3 * area
peak_time = 1e-3
applied_impulse = peak_time * peak_force / 2
peak_ini = 1
peak_end = peak_ini + np.searchsorted(t_vector >= peak_time, True)
force_up = 0 * t_vector[0:peak_ini]
force_down = peak_force - (peak_force / peak_time) * (t_vector[0:peak_end - peak_ini])
force_vector = np.hstack((force_up, force_down, np.zeros(t_N - peak_end)))
load_dof = n_dof - 1
print(f"{peak_force=}")
print(f"{applied_impulse=}")

# Constraints
fixed_dof = 0

# Initial conditions
dof0 = n_dof - 1
d0 = 0.0
v0 = 0.0

maxiter = 500
flags = {'opt_uniform': True,
         'opt_fg': True,
         'method': 'differential_evolution',  # 'simplex',  #
         'disp': True,
         'workers': 8}
opt_id = '04'

if __name__ == '__main__':
    mesh = (cl.Mesh(total_length, n_dof))
    mesh.fill_elements('k', k) \
        .fill_elements('penalty_gap', penalty_gap) \
        .fill_elements('c', c) \
        .fill_elements('m', m)
    # .fill_elements('muN', muN) \
    constraints = cl.Constraint(dof_s=fixed_dof)
    loads = cl.Load(dof_s=load_dof, t=t_vector, force=force_vector)
    initial_conditions = [cl.InitialCondition(dof0, 'displacement', d0),
                          cl.InitialCondition(dof0, 'velocity', v0)]
    op = {'t_vector': t_vector, 'method': 'BDF'}
    model = cl.Model(mesh=mesh, constraints=constraints, loads=loads, initial_conditions=initial_conditions, options=op)

    t_i = datetime.datetime.now()
    model.solve()
    print("Elapsed time:", datetime.datetime.now() - t_i)

    # BASE
    lb, ub = bounds.bounds(min_rel=min_rel, max_rel=max_rel, c=c, m=m)
    opt_uniform = optim.Optimization(base_model=model.deepcopy(), obj_fun=obj_fun_peak, lb=lb, ub=ub, uniform=True)
    print('base:', opt_uniform.opt_obj_func())
    print('with', [element.props['value'] for element in opt_uniform.model.mesh.elements])
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
    print('uniform optimal:', opt_uniform.opt_obj_func())
    print('with', [element.props['value'] for element in opt_uniform.model.mesh.elements])
    _, axs = plt.subplots(4, 1, sharex='all')
    plot_results(axs, t=opt_uniform.model.sol.t,
                 f_reaction=opt_uniform.model.reactions(fixed_dof), impulse=opt_uniform.model.impulses(fixed_dof),
                 d=opt_uniform.model.displacements(dof0), v=opt_uniform.model.velocities(dof0),
                 t_load=t_vector, f_load=force_vector, label='base')
    fig, ax, ani = opt_uniform.model.animate()
    plt.show()

    # FUNCTIONALLY GRADED
    if flags['opt_fg']:
        for mass_element in [f"m_{i}_{i + 1}" for i in range(n_dof - 1)]:
            ub[mass_element] *= (n_dof-1)
        opt_fg = optim.Optimization(base_model=opt_uniform.model.deepcopy(), obj_fun=obj_fun_peak, lb=lb, ub=ub,
                                    uniform=False, opt_obj_fun_override=opt_obj_fun_override)
        opt_fg.optimize(maxiter=maxiter, disp=flags['disp'], method=flags['method'], workers=flags['workers'])
        with open(f'opt_fg_{opt_id}.pk', 'wb') as file:
            pickle.dump(opt_fg, file)
    else:
        with open(f'opt_fg_{opt_id}.pk', 'rb') as file:
            opt_fg = pickle.load(file)
    print('functionally graded optimal:', opt_fg.opt_obj_func())
    print('with', [element.props['value'] for element in opt_fg.model.mesh.elements])
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

...
