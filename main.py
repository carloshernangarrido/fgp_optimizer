from models import chain_like as cl


n_dof = 10
length = 10
total_mass = 1
k = 10
c = .1
muN = 5
gap = {'value': 0.1*(length/(n_dof-1)),
       'contact_stiffness': 10*k}

if __name__ == '__main__':
    mesh = (cl.Mesh(length, n_dof, total_mass))
    mesh.fill_elements('k', k).fill_elements('gap', {'value': .3, 'contact_stiffness': 33.})
    const = cl.Constraint(0)
    model = cl.Model(mesh, const)
...

