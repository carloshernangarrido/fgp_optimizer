import numpy as np
from models import chain_like as cl
import scipy as sp


n_dof = 10
length = 10
total_mass = 1
k = 10
c = .1
muN = 5
gap = {'value': 0.1*(length/(n_dof-1)),
       'contact_stiffness': 10*k}

t_vector = np.linspace(0, 10, 100)
force_vector = 0*t_vector


if __name__ == '__main__':
    mesh = (cl.Mesh(length, n_dof, total_mass))
    mesh.fill_elements('k', k).fill_elements('gap', {'value': .3, 'contact_stiffness': 33.})
    const = cl.Constraint(dof_s=0)
    load = cl.Load(dof_s=n_dof+1, t=t_vector, force=force_vector)
    model = cl.Model(mesh, const)
    # sp.integrate.solve_ivp(model.f, )
...
