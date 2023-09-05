
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

# Weigth for deformations
max_def_restriction = .020
lambda_ = 1e3*140000/max_def_restriction  # peak
# lambda_ = 1e3*100/max_def_restriction  # impulse

# Loading
scale = 1.0
times_t = 3
subsampling = 10
loading_path = r"C:\Users\joses\Mi unidad\TRABAJO\48_FG_protection\TRABAJO\loadings"
loading_filename = r"Ident 0 - 2d-1-5kg-0-52m01.uhs"
loading_df = pd.read_csv(os.path.join(loading_path, loading_filename), skiprows=2)
loading_df = loading_df.loc[:, (loading_df.columns.str.strip() == 'TIME (ms)') |
                               (loading_df.columns.str.strip() == 'PRESSURE')]
loading_df.loc[:, loading_df.columns.str.strip() == 'TIME (ms)'] = .001 * loading_df.loc[:,
                                                                          loading_df.columns.str.strip() == 'TIME (ms)']
loading_df.loc[:, loading_df.columns.str.strip() == 'PRESSURE'] = 1000 * loading_df.loc[:,
                                                                         loading_df.columns.str.strip() == 'PRESSURE']
loading_df.columns = ('time', 'pressure')
loading_df.loc[:, 'pressure'] = loading_df.loc[:, 'pressure'] - loading_df.loc[0, 'pressure']
force_vector = area * loading_df['pressure'].values
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
element_length = .024  # m
total_length = element_length * n_elements

min_rel, max_rel = 0.001, 1.00
k, c, gap, kc = kcgapkc_from_al(area=area, length=element_length, material='characterized_viscoelastic_foam')
# k, c, gap, kc = kcgapkc_from_al(area=area, length=element_length, material='c100times_characterized_viscoelastic_foam')

k = 100*k
c = 100*c

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

maxiter = 100
flags = {'obj_fun': 'peak',  # 'peak', 'impulse'
         'fun_override': 'denskc_m',  # 'density', 'denskc_m'
         'opt_uniform': True,
         'opt_fg': True,
         'method': 'differential_evolution',  # 'simplex',  #
         'disp': True,
         'workers': 7}
opt_id = '00'
