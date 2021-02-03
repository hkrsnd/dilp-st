def get_parameters(name):
    if name == 'member':
        N_beam = 3
        T_beam = 3
        T = 4
        m = 2
    elif name == 'subtree':
        N_beam = 15
        T_beam = 3
        T = 6
        m = 3
    elif name == 'append':
        N_beam = 10
        T_beam = 5
        T = 4
        m = 3
    elif name == 'plus':
        N_beam = 10
        T_beam = 5
        T = 8
        m = 3
    elif name == 'delete':
        N_beam = 10
        T_beam = 5
        T = 4
        m = 2

    return N_beam, T_beam, T, m
