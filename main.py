import datetime
import timeit
import numpy as np
import matplotlib.pyplot as plt
from models import chain_like as cl


# Time
t_ini = 0.0
t_fin = 5
delta_t = .01
t_vector = np.arange(t_ini, t_fin, delta_t)
t_N = len(t_vector)

n_dof = 5
length = 10
zeta = .01
total_mass = .1
total_stiffness = .01
m = total_mass / (n_dof-1)
k = total_stiffness * (n_dof-1)
c = zeta * (2 * np.sqrt(k * m))

# Plastification
muN = .5
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
    model = cl.Model(mesh=mesh, constraints=constraints, loads=loads, initial_conditions=initial_conditions)

    t_i = datetime.datetime.now()
    sol = model.solve(t_vector=t_vector, method='BDF')
    print("Elapsed time:", datetime.datetime.now() - t_i)

    _, axs = plt.subplots(3, 1, sharex='all')
    axs[1].set_ylabel('force')
    axs[0].plot(sol.t, model.reactions(fixed_dof), label='reaction')
    axs[0].plot(t_vector, force_vector, label='load')
    axs[1].set_ylabel('displacement')
    axs[1].plot(sol.t, model.displacements(dof0))
    axs[2].set_ylabel('velocities')
    axs[2].plot(sol.t, model.velocities(dof0))
    axs[2].set_xlabel('time')
    plt.show()

    fig, ax, ani = model.animate()

    plt.show()
...
