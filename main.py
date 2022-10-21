import datetime
import numpy as np
import matplotlib.pyplot as plt

from models import chain_like as cl
from optimization import optimizers as optim


# Time
t_ini = 0.0
t_fin = 5
delta_t = .1
t_vector = np.arange(t_ini, t_fin, delta_t)
t_N = len(t_vector)

n_dof = 5
length = 10
zeta = .01
total_mass = 1.0
total_stiffness = .01
m = total_mass / (n_dof-1)
k = total_stiffness * (n_dof-1)
c = zeta * (2 * np.sqrt(k * m))

# Plastification
muN = 0.5
element_length = length/(n_dof-1)
displacement_tol = .01
muN = {'value': muN,
       'v_th': element_length*displacement_tol/(t_fin-t_ini)}
# Compaction
gap = {'value': 0.9 * (length / (n_dof - 1)),
       'contact_stiffness': 10000 * k}


# Load: triangular pulse
peak_force = 1
peak_time = 1
peak_i = np.searchsorted(t_vector >= peak_time, True)
force_up = (peak_force/peak_time) * t_vector[0:peak_i]
force_down = peak_force - (peak_force/peak_time) * t_vector[0:peak_i]
force_vector = np.hstack((force_up, force_down, np.zeros(t_N - 2*peak_i)))

load_dof = n_dof - 1

# Constraints
fixed_dof = 0

# Initial conditions
dof0 = n_dof - 1
d0 = 0.0
v0 = 0.0

optimize_flag = True

if __name__ == '__main__':
    mesh = (cl.Mesh(length, n_dof, total_mass))
    mesh.fill_elements('k', k) \
        .fill_elements('c', c) \
        .fill_elements('muN', muN) \
        .fill_elements('gap', gap)
    constraints = cl.Constraint(dof_s=fixed_dof)
    loads = cl.Load(dof_s=load_dof, t=t_vector, force=force_vector)
    initial_conditions = [cl.InitialCondition(dof0, 'displacement', d0),
                          cl.InitialCondition(dof0, 'velocity', v0)]
    op = {'t_vector': t_vector, 'method': 'BDF'}
    model = cl.Model(mesh=mesh, constraints=constraints, loads=loads, initial_conditions=initial_conditions, options=op)

    t_i = datetime.datetime.now()
    model.solve()
    print("Elapsed time:", datetime.datetime.now() - t_i)

    # fig, ax, ani = model.animate()
    # plt.show()

    if optimize_flag:
        # Optimization
        def obj_fun(model_: cl.Model):
            return max(abs(model_.reactions(fixed_dof)))

        lb = {'muN_0_1': 0.1 * muN['value'],
              'muN_1_2': 0.1 * muN['value'],
              'muN_2_3': 0.1 * muN['value'],
              'muN_3_4': 0.1 * muN['value']}
        ub = {'muN_0_1': 10. * muN['value'],
              'muN_1_2': 10. * muN['value'],
              'muN_2_3': 10. * muN['value'],
              'muN_3_4': 10. * muN['value']}
        optimization = optim.Optimization(base_model=model, obj_fun=obj_fun, lb=lb, ub=ub)

        base_result_value = optimization.opt_obj_func()
        print(base_result_value)
        opt_result = optimization.optimize(maxiter=100, disp=True)
        print(opt_result)

        _, axs = plt.subplots(3, 1, sharex='all')
        # base design
        axs[0].set_ylabel('force')
        axs[0].plot(model.sol.t, model.reactions(fixed_dof), label='reaction')
        axs[0].plot(t_vector, force_vector, label='load')
        axs[1].set_ylabel('displacement')
        axs[1].plot(model.sol.t, model.displacements(dof0))
        axs[2].set_ylabel('velocities')
        axs[2].plot(model.sol.t, model.velocities(dof0))
        axs[2].set_xlabel('time')
        # optimized design
        axs[0].set_ylabel('force')
        axs[0].plot(optimization.model.sol.t, optimization.model.reactions(fixed_dof), label='reaction')
        axs[0].plot(t_vector, force_vector, label='load')
        axs[1].set_ylabel('displacement')
        axs[1].plot(optimization.model.sol.t, optimization.model.displacements(dof0))
        axs[2].set_ylabel('velocities')
        axs[2].plot(optimization.model.sol.t, optimization.model.velocities(dof0))
        axs[2].set_xlabel('time')
        plt.show()
    else:
        _, axs = plt.subplots(3, 1, sharex='all')
        # base design
        axs[0].set_ylabel('force')
        axs[0].plot(model.sol.t, model.reactions(fixed_dof), label='reaction')
        axs[0].plot(t_vector, force_vector, label='load')
        axs[1].set_ylabel('displacement')
        axs[1].plot(model.sol.t, model.displacements(dof0))
        axs[2].set_ylabel('velocities')
        axs[2].plot(model.sol.t, model.velocities(dof0))
        axs[2].set_xlabel('time')
        plt.show()

...
