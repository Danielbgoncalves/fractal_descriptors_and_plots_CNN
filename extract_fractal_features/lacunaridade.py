import numpy as np

def lacunaridade(p):
    '''
    Calcula a lacunaridade de uma matriz p 
    por escala, ou seja, para cada r coluna de p 
    '''
    # lac = np.zeros(p.shape[1])

    # for i in range(p.shape[1]):
    #     m1, m2 = 0, 0
    #     for j in range(p.shape[0]):
    #         m1 += (j+1) * p[j, i]
    #         m2 += (j+1) * (j+1) * p[j, i]

    #     lac.append( (m2-(np.power(m1,2) )) / np.power(m1,2) )

    maxR, _ = p.shape

    j = np.arange(1, maxR+1).reshape(-1,1)

    m1 = np.sum(j * p, axis=0)
    m2 = np.sum((j ** 2) * p, axis=0)

    lac = (m2 - (m1 ** 2)) / (m1 ** 2) 

    return lac.tolist()

def lacunaridade_plus(p):

    maxR = p.shape[0]

    j = np.arange(1, maxR+1, dtype=np.float64)

    m1 = j @ p
    m2 = (j ** 2) @ p

    m1_sq = m1 ** 2
    lac = (m2 - m1_sq) / m1_sq 

    return lac.tolist()