import numpy as np
from numba import njit, prange

@njit
def pmrEucl(img, maxr):
    '''
    Calcula a matriz de probabilidades de uma imagem usano distância eclidiana 
    img deve ser uma ndarray numpy convertido em uint8 para float64
    maxr é o limite superior do rio, deve ser impar.
    '''
    aux = img.astype(np.float64)
    r = list(range(3, maxr + 1, 2))
    p = np.zeros((r[-1]**2, len(r)), dtype=np.float64)

    for k in range(len(r)):
        rk = r[k]
        ncaixas = (img.shape[0] - rk + 1) * (img.shape[1] - rk + 1)
        lim = (rk / 2) - 0.5

        for x in range(int(lim), img.shape[0] - int(lim)):
            for y in range(int(lim), img.shape[1] - int(lim)):
                m = 0
                xi = int( x - lim )
                xf = int( x + lim )
                yi = int( y - lim )
                yf = int( y + lim )

                for i in range(xi, xf + 1):
                    for j in range(yi, yf + 1):
                        dist = np.sqrt(
                            (aux[i, j, 0] - aux[x, y, 0])**2 + 
                            (aux[i, j, 1] - aux[x, y, 1])**2 + 
                            (aux[i, j, 2] - aux[x, y, 2])**2 
                        )
                        if dist <= rk:
                            m += 1
                p[m-1,k] += 1
        p[:, k] = p[:, k] / ncaixas

    return p

@njit(parallel=True)
def pmrEucl_plus(img, maxr=41):
    aux = img.astype(np.float64)
    # Numba lida melhor com arrays do que com listas geradas por range()
    r = np.arange(3, maxr+1, 2) 
    max_m = r[-1]**2 + 1
    
    p = np.zeros((max_m, len(r)), dtype=np.float64)

    # para cada tamanho de caixa (não podemos paralelizar aqui por desbalanceamento)
    for k in range(len(r)):
        ncaixas = float((img.shape[0] - r[k]+1) * (img.shape[1] - r[k]+1))
        lim = (r[k]/2) - 0.5
        lim_int = int(lim)

        r_k2 = r[k]**2

        # CRIANDO A GAVETA TEMPORÁRIA:
        # Cada thread vai escrever APENAS na linha 'x' deste p_temp
        p_temp = np.zeros((img.shape[0], max_m), dtype=np.float64)

        # percorrer os pixels centrais em PARALELO
        for x in prange(lim_int, img.shape[0] - lim_int):
            for y in range(lim_int, img.shape[1] - lim_int):
                m = 0
                xi = int( x - lim )
                xf = int( x + lim )
                yi = int( y - lim )
                yf = int( y + lim )

                # deslizar a caixa
                for i in range(xi, xf + 1):
                    for j in range(yi, yf + 1):
                        # Checagem Eucl

                        dist_r = aux[i, j, 0] - aux[x, y, 0]
                        dist_g = aux[i, j, 1] - aux[x, y, 1]
                        dist_b = aux[i, j, 2] - aux[x, y, 2]
                        
                        if ( dist_r**2 + dist_g**2 + dist_b**2 <= r_k2):
                            m += 1
                                    
             
                p_temp[x, m] += 1
                
        for m_idx in range(max_m):
            soma = 0.0
            for x_idx in range(img.shape[0]):
                soma += p_temp[x_idx, m_idx]
            p[m_idx, k] = soma / ncaixas

    return p[1:, :]