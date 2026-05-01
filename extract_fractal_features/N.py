import numpy as np

def N(matriz):
    '''
    Recebe uma matriz de probabilidade em que cada coluna j representa um raio r
    e cada linha i representa a quantidade de pixels similares ao central da caixa. 
    Retorna um vetor com a média, o inverso da massa de pixels.
    '''
    
    # Tradução direta
    # MaxR, MaxC = matriz.shape
    # NL = np.array((MaxR, MaxC))

    # for i in range(MaxR):
    #     for j in range(MaxC):
    #         NL[i][j] = matriz[i][j]/ (i+1)

    # NLf = np.sum(NL, axis=0)

    # Tradução numpy
    maxR, _ = matriz.shape

    divisores = np.array(range(1,maxR+1)).reshape(-1,1) # [1,2,3..] vira coluna 
    NL = matriz / divisores
    NLf = np.sum(NL, axis=0)
    
    return NLf.tolist()