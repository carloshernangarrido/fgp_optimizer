"""
input parameters for 3 visco-elastic foam plates separated by steel plates.
foam plate area = 0.3 m x 0.3 m
foam plate thickness = 0.024 m
steel plate thickness = (1/8, 1/4, 3/8) x 0.0254 m
"""

from material_properties import kc_from_al, m_from_al
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load loading file
loading_path = r"C:\Users\joses\Mi unidad\TRABAJO\48_FG_protection\TRABAJO\loadings"
loading_filename = r"Ident 0 - 2d-1-5kg-0-52m01.uhs"
loading_df = pd.read_csv(os.path.join(loading_path, loading_filename), skiprows=2)
loading_df = loading_df.loc[:, (loading_df.columns.str.strip() == 'TIME (ms)') |
                               (loading_df.columns.str.strip() == 'PRESSURE')]
loading_df.loc[:, loading_df.columns.str.strip() == 'TIME (ms)'] = .001*loading_df.loc[:,
                                                                        loading_df.columns.str.strip() == 'TIME (ms)']
loading_df.loc[:, loading_df.columns.str.strip() == 'PRESSURE'] = 1000*loading_df.loc[:,
                                                                       loading_df.columns.str.strip() == 'PRESSURE']
loading_df.columns = ('time', 'pressure')
loading_df.loc[:, 'pressure'] = loading_df.loc[:, 'pressure'] - loading_df.loc[0, 'pressure']
plt.plot(loading_df['time'], loading_df['pressure'])
plt.xlabel('time (s)')
plt.ylabel('pressure (Pa)')
plt.show()

# Time
t_ini = loading_df.iloc[0]['time']
t_fin = loading_df.iloc[-1]['time']
t_vector = loading_df['time'].values

area = 0.3 * 0.3  # m^2
n_dof = 4
element_length = .24  # m
total_length = element_length * (n_dof - 1)

min_rel, max_rel = 0.50, 1.00
k, c = kc_from_al(area=area, length=element_length, material='viscoelastic_foam')
m = m_from_al(area=area, length=0.001, material='lead')
max_mass = m * max_rel * (n_dof - 1)
n_elements = n_dof-1

# Plastification
# muN = .06
displacement_tol = .01
# muN = {'value': muN,
#        'v_th': element_length*displacement_tol/(t_fin-t_ini)}
# Compaction
gap = {'value': 0.75 * element_length,
       'contact_stiffness': 10000 * k}
penalty_gap = {'value': 0.75 * element_length,
               'contact_stiffness': 10000 * k,
               'penetration': 0.01 * 0.75 * element_length}

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
opt_id = '06'