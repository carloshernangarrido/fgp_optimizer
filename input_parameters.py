"""
input parameters for 3 visco-elastic foam plates separated by steel plates.
foam plate area = 0.3 m x 0.3 m
foam plate thickness = 0.024 m
steel plate thickness = 3/8 x 0.0254 m
"""
import numpy as np

from material_properties import kcgapkc_from_al, m_from_al
import os
import pandas as pd
import matplotlib.pyplot as plt

scale = 1.0

n_dof = 4
area = 0.3 * 0.3  # m^2

# Loading
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
force_vector = scale * np.hstack((force_vector, force_vector[-1] + 0 * force_vector[1:-1]))
# time
t_ini = loading_df.iloc[0]['time']
t_fin = loading_df.iloc[-1]['time']
t_vector = loading_df['time'].values
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

# Protection design
n_elements = n_dof - 1
load_dof = n_dof - 1
element_length = .024  # m
total_length = element_length * n_elements

min_rel, max_rel = 0.50, 1.00
k, c, gap, kc = kcgapkc_from_al(area=area, length=element_length, material='characterized_viscoelastic_foam')
m = m_from_al(area=area, length=(3 / 8) * .0254, material='steel')
min_mass_dof = 0.01 * m
max_mass_dof = m
max_mass_total = m

# Compaction
penalty_gap = {'value': gap,
               'contact_stiffness': kc,
               'penetration': 0.01 * gap}

k_with_failure = {'value': k,
                  'allowable_deformation': 0.9 * element_length}

# Constraints
fixed_dof = 0

# Initial conditions
dof0 = n_dof - 1
d0 = 0.0
v0 = 0.0

animate_each = 5

maxiter = 100
flags = {'opt_uniform': True,
         'opt_fg': True,
         'method': 'differential_evolution',  # 'simplex',  #
         'disp': True,
         'workers': 8}
opt_id = '00'
