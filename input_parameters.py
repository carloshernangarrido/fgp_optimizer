
"""
Input parameters for 3 visco-elastic foam plates separated by steel plates.
foam plate area = 0.3 m x 0.3 m
foam plate thickness = 0.024 m
steel plate thickness = 1/4 x 0.0254 m
"""
import numpy as np

from material_properties import kcgapkc_from_al, m_from_al
import os
import pandas as pd
import matplotlib.pyplot as plt


n_dof = 5
area = 0.3 * 0.3  # m^2

max_def_restriction_m024 = .020

# Loading
radial_integration_flag = True
radial_increment = 0.050
scale = 1.0
times_t = 2
subsampling = 10
loading_path = r"C:\Users\joses\Mi unidad\TRABAJO\48_FG_protection\TRABAJO\loadings"
loading_filename = r"Ident 0 - 2d-1-5kg-0-52m01.uhs"
loading_df = pd.read_csv(os.path.join(loading_path, loading_filename), skiprows=2)
if not radial_integration_flag:
    loading_df = loading_df.loc[:, (loading_df.columns.str.strip() == 'TIME (ms)') |
                                   (loading_df.columns.str.strip() == 'PRESSURE')]
    loading_df.loc[:, loading_df.columns.str.strip() == 'TIME (ms)'] = .001 * loading_df.loc[:,
                                                                              loading_df.columns.str.strip() == 'TIME (ms)']
    loading_df.loc[:, loading_df.columns.str.strip() == 'PRESSURE'] = 1000 * loading_df.loc[:,
                                                                             loading_df.columns.str.strip() == 'PRESSURE']
    loading_df.columns = ('time', 'pressure')
    loading_df.loc[:, 'pressure'] = loading_df.loc[:, 'pressure'] - loading_df.loc[0, 'pressure']

    force_vector = area * loading_df['pressure'].values
else:
    loading_df = loading_df.iloc[:, 1:]
    loading_df.loc[:, loading_df.columns.str.strip() == 'TIME (ms)'] = .001 * loading_df.loc[:,
                                                                              loading_df.columns.str.strip() == 'TIME (ms)']
    for i in range(1, len(loading_df.columns)):
        loading_df.iloc[:, i] = 1000 * loading_df.iloc[:, i]
    loading_df.columns = ['time', ] + [f'pressure.{i}' for i in range(len(loading_df.columns) - 1)]

    # Equivalent radius (area = pi r^2)
    r_max = np.sqrt(area / np.pi)
    i_max = int(1+(r_max/radial_increment))

    radial_positions = [0.050 * _ for _ in range(i_max)]
    force_vector = 0.0*loading_df['time'].values.reshape((-1, 1))
    for i in range(1, len(radial_positions)):
        p1 = loading_df.iloc[:, i].values.reshape((-1, 1))
        p2 = loading_df.iloc[:, i + 1].values.reshape((-1, 1))
        r1 = radial_positions[i - 1]
        r2 = radial_positions[i]
        kp = (p2-p1)/(r2-r1)
        force_vector += (p1 - kp*r1)*np.pi*(r2**2 - r1**2) + kp * (2/3) * np.pi* (r2**3 - r1**3)
    force_vector = force_vector.reshape((-1,))
# plt.plot(force_vector)
# plt.show()

# time
t_ini = loading_df.iloc[0]['time']
t_fin = loading_df.iloc[-1]['time']
t_vector = loading_df['time'].values

for i in range(times_t):
    force_vector = scale * np.hstack((force_vector, 0 * force_vector[1:-1]))
    t_vector = np.hstack((t_vector, t_vector[1:-1] + t_vector[-1]))

if subsampling is not None:
    t_vector = t_vector[0::subsampling]
    force_vector = force_vector[0::subsampling]
# plots
# _, axs = plt.subplots(2, 1)
# axs[0].plot(loading_df['time'], loading_df['pressure'])
# axs[0].set_ylabel('pressure (Pa)')
# axs[1].plot(t_vector, force_vector)
# axs[1].set_ylabel('force (N)')
# axs[1].set_xlabel('time (s)')
# plt.show()
peak_force = max(force_vector)
applied_impulse = np.trapz(y=force_vector, x=t_vector)

# Protected structure
protected_structure = {'m': 57.6*2/np.pi, 'k': (57.6*2/np.pi)*(2*np.pi*120)**2, 'c': 0}

# Protection design
n_elements = n_dof - 1
load_dof = n_dof - 1
element_length = .024 * 3/(n_elements-1)  # m
total_length = element_length * n_elements
max_def_restriction = max_def_restriction_m024 * 3/(n_elements-1)

min_rel, max_rel = 0.01, 1.00
k, c, gap, kc = kcgapkc_from_al(area=area, length=element_length, material='characterized_viscoelastic_foam')
# k, c, gap, kc = kcgapkc_from_al(area=area, length=element_length, material='c100times_characterized_viscoelastic_foam')

m = m_from_al(area=area, length=(1/4) * .0254, material='steel')
min_mass_dof = 0.001 * m
max_mass_dof = m
max_mass_total = m

# Compaction
penalty_gap = {'value': gap,
               'contact_stiffness': kc,
               'penetration': 0.01 * gap}

k_with_failure = {'value': k,
                  'allowable_deformation': 1.0 * element_length}

# Constraints
fixed_dof = 0

# Initial conditions
dof0 = load_dof
d0 = 0.0
v0 = 0.0

animate_each = 5

# Weigth for deformations
lambda_ = 1e3*140000/max_def_restriction  # peak
# lambda_ = 1e3*100/max_def_restriction  # impulse

# optimization options
maxiter = 1000
obj_fun_scaling = 10e-8


flags = {'obj_fun': 'peak',  # 'peak', 'impulse'
         'fun_override': 'density',  # 'density', 'denskc_m'
         'lumped_masses': True,
         'opt_uniform': True,
         'opt_fg': True,
         'method': 'differential_evolution',  #'simplex',  #
         'disp': True,
         'workers': 8}

if flags['fun_override'] == 'denskc_m':
    k = 10 * k
    c = 10 * c

opt_id = '05'
