import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt

from material_properties import kc_from_al
from models import chain_like as cl
from optimization import optimizers as optim
from plotting.results import plot_results

# Time
t_ini = 0.0
t_fin = .05
delta_t = .0001
t_vector = np.arange(t_ini, t_fin, delta_t)
t_N = len(t_vector)

area = 0.1*0.1
n_dof = 5
total_length = 0.05
element_length = total_length/(n_dof-1)
# zeta = .01
total_mass = 1.0
# total_stiffness = .001
m = total_mass / (n_dof - 1)
# k = total_stiffness * (n_dof - 1)
# c = zeta * (2 * np.sqrt(k * m))
# c = 0.1
k, c = kc_from_al(area=area, length=element_length)
k *= 1

# Plastification
# muN = .06
displacement_tol = .01
# muN = {'value': muN,
#        'v_th': element_length*displacement_tol/(t_fin-t_ini)}
# Compaction
gap = {'value': 0.75 * element_length,
       'contact_stiffness': 10000 * k}

# Load: triangular pulse
peak_force = 200e3 * area
peak_time = 1e-3
applied_impulse = peak_time*peak_force/2
print(f"{applied_impulse=}")
peak_ini = 1
peak_end = peak_ini + np.searchsorted(t_vector >= peak_time, True)
force_up = 0 * t_vector[0:peak_ini]
force_down = peak_force - (peak_force / peak_time) * (t_vector[0:peak_end - peak_ini])
force_vector = np.hstack((force_up, force_down, np.zeros(t_N - peak_end)))

load_dof = n_dof - 1

# Constraints
fixed_dof = 0

# Initial conditions
dof0 = n_dof - 1
d0 = 0.0
v0 = 0.0

maxiter = 10
flags = {'opt_uniform': True,
         'opt_fg': True,
         'method': 'differential_evolution',  # 'simplex',  #
         'disp': True}
opt_id = '01'

if __name__ == '__main__':
    mesh = (cl.Mesh(total_length, n_dof, total_mass=0))
    mesh.fill_elements('k', k) \
        .fill_elements('gap', gap) \
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


    def obj_fun(model_: cl.Model):
        return max(abs(model_.reactions(fixed_dof)))
        # return max(abs(model_.impulses(fixed_dof)))

    min_rel, max_rel = .1, 10
    lb = {'c_0_1': min_rel * c,
          'c_1_2': min_rel * c,
          'c_2_3': min_rel * c,
          'c_3_4': min_rel * c}
    ub = {'c_0_1': max_rel * c,
          'c_1_2': max_rel * c,
          'c_2_3': max_rel * c,
          'c_3_4': max_rel * c}
    # lb = {'muN_0_1': min_rel * muN['value'],
    #       'muN_1_2': min_rel * muN['value'],
    #       'muN_2_3': min_rel * muN['value'],
    #       'muN_3_4': min_rel * muN['value']}
    # ub = {'muN_0_1': max_rel * muN['value'],
    #       'muN_1_2': max_rel * muN['value'],
    #       'muN_2_3': max_rel * muN['value'],
    #       'muN_3_4': max_rel * muN['value']}
    # lb = {'m_0_1': min_rel * m,
    #       'm_1_2': min_rel * m,
    #       'm_2_3': min_rel * m,
    #       'm_3_4': min_rel * m}
    # ub = {'m_0_1': max_rel * m,
    #       'm_1_2': max_rel * m,
    #       'm_2_3': max_rel * m,
    #       'm_3_4': max_rel * m}
    # BASE
    opt_uniform = optim.Optimization(base_model=model.deepcopy(), obj_fun=obj_fun, lb=lb, ub=ub, uniform=True)
    print('base:', opt_uniform.opt_obj_func())
    _, axs = plt.subplots(4, 1, sharex='all')
    plot_results(axs, t=model.sol.t, f_reaction=model.reactions(fixed_dof), impulse=model.impulses(fixed_dof),
                 d=model.displacements(dof0), v=model.velocities(dof0),
                 t_load=t_vector, f_load=force_vector, label='base')
    plt.show()
    fig, ax, ani = model.animate()
    plt.show()

    # UNIFORM
    if flags['opt_uniform']:
        opt_uniform.optimize(maxiter=maxiter, disp=flags['disp'], method=flags['method'])
        with open('opt_uniform.pk', 'wb') as file:
            pickle.dump(opt_uniform, file)
    else:
        with open('opt_uniform.pk', 'rb') as file:
            opt_uniform = pickle.load(file)
    print('uniform optimal:', opt_uniform.opt_obj_func())
    _, axs = plt.subplots(4, 1, sharex='all')
    plot_results(axs, t=opt_uniform.model.sol.t,
                 f_reaction=opt_uniform.model.reactions(fixed_dof), impulse=opt_uniform.model.impulses(fixed_dof),
                 d=opt_uniform.model.displacements(dof0), v=opt_uniform.model.velocities(dof0),
                 t_load=t_vector, f_load=force_vector, label='base')
    plt.show()
    fig, ax, ani = opt_uniform.model.animate()
    plt.show()

    # FUNCTIONALLY GRADED
    if flags['opt_fg']:
        opt_fg = optim.Optimization(base_model=opt_uniform.model.deepcopy(), obj_fun=obj_fun, lb=lb, ub=ub,
                                    uniform=False)
        opt_fg.optimize(maxiter=maxiter, disp=flags['disp'], method=flags['method'])
        with open(f'opt_fg_{opt_id}.pk', 'wb') as file:
            pickle.dump(opt_fg, file)
    else:
        with open(f'opt_fg_{opt_id}.pk', 'rb') as file:
            opt_fg = pickle.load(file)
    print('functionally graded optimal:', opt_fg.opt_obj_func())
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
    plt.show()
    fig, ax, ani = opt_fg.model.animate()
    plt.show()

...
