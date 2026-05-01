import numpy as np
from N import N

def D(p):
    '''
    Recebe uma matriz p proveniente de alguma função pmr
    retorna em d o valor da dimensão fractal calculado para cada raio r
    '''
    num_col = p.shape[1]
    d = np.zeros(num_col)

    if num_col > 1:
        r = np.zeros(num_col)
        for i in range(num_col):
            r[i] = 3 + ((i+1)*2 - 2)
    else:
        r = np.sqrt(p.shape[0])
    
    for i in range(num_col):
        d[i] = -( np.log(N(p[:,i])) / np.log(r[i]) )
    
    return d