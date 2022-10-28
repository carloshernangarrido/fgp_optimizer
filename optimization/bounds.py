

def bounds(min_rel: float = .01, max_rel: float = 1,
           c: float = None, m: float = None, muN: dict = None):
    if c is not None and m is not None:
        lb = {'c_0_1': min_rel * c,
              'c_1_2': min_rel * c,
              'c_2_3': min_rel * c,
              'c_3_4': min_rel * c,
              'm_0_1': min_rel * m,
              'm_1_2': min_rel * m,
              'm_2_3': min_rel * m,
              'm_3_4': min_rel * m}
        ub = {'c_0_1': max_rel * c,
              'c_1_2': max_rel * c,
              'c_2_3': max_rel * c,
              'c_3_4': max_rel * c,
              'm_0_1': max_rel * m,
              'm_1_2': max_rel * m,
              'm_2_3': max_rel * m,
              'm_3_4': max_rel * m}
    elif muN is not None:
        lb = {'muN_0_1': min_rel * muN['value'],
              'muN_1_2': min_rel * muN['value'],
              'muN_2_3': min_rel * muN['value'],
              'muN_3_4': min_rel * muN['value']}
        ub = {'muN_0_1': max_rel * muN['value'],
              'muN_1_2': max_rel * muN['value'],
              'muN_2_3': max_rel * muN['value'],
              'muN_3_4': max_rel * muN['value']}
    elif m is not None:
        lb = {'m_0_1': min_rel * m,
              'm_1_2': min_rel * m,
              'm_2_3': min_rel * m,
              'm_3_4': min_rel * m}
        ub = {'m_0_1': max_rel * m,
              'm_1_2': max_rel * m,
              'm_2_3': max_rel * m,
              'm_3_4': max_rel * m}
    else:
        raise NotImplementedError
    return lb, ub
