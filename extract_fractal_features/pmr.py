import numpy as np
from numba import njit, prange
# from numpy.lib.stride_tricks import sliding_window_view


@njit
def pmr(img, maxr=41):
    '''
    Calcula a matriz de probabilidades de uma imagem
    img deve ser uma ndarray numpy convertido em uint8 para float64
    maxr é o limite superior do rio, deve ser impar.
    '''

    '''
    Tem como vetorizar em numpy pra deixar mais rápido
    '''

    aux = img.astype(np.float64)
    r = list(range(3, maxr+1, 2)) #[3, 5, 7, ... maxr]
    p = np.zeros((r[-1]**2 + 1, len(r)), dtype=np.float64)

    # para cada tamanho de caixa
    for k in range(len(r)):
        ncaixas = float((img.shape[0] - r[k]+1) * (img.shape[1] - r[k]+1))
        lim = (r[k]/2) - 0.5

        # percorrer os pixels centrais
        for x in range(int(lim), img.shape[0] - int(lim)):
            for y in range(int(lim), img.shape[1] - int(lim)):
                m = 0
                xi = int( x - lim )
                xf = int( x + lim )
                yi = int( y - lim )
                yf = int( y + lim )

                # deslizar a caixa
                for i in range(xi, xf + 1):
                    for j in range(yi, yf + 1):
                        dist = abs(aux[i,j,0] - aux[x,y,0])
                        if dist <= r[k]:
                            dist = abs(aux[i,j,1] - aux[x,y,1])
                            if dist <= r[k]:
                                dist = abs(aux[i,j,2] - aux[x,y,2])
                                if dist <= r[k]:
                                    m += 1
                p[m,k] += 1
        p[:,k] = p[:,k] / ncaixas

    return p[1:, :] 



    # Inviável -> estouro de memória, essa bomba ta querendo alocar 2G na stack, calma lá ne plmd
    # aux = img.astype(np.float32)
    # r = list(range(3, maxr+1, 2))  # [3, 5, 7, ...]
    # p = np.zeros((r[-1]**2 + 1, len(r)), dtype=np.float64)

    # for k, rk in enumerate(r):
    #     lim = (rk / 2) - 0.5
    #     lim_i = int(lim)
    #     ncaixas = float((img.shape[0] - rk + 1) * (img.shape[1] - rk + 1))

    #     jan = sliding_window_view(aux, (rk, rk, 1))[..., 0, :, :, :]

    #     centros = aux[lim_i:-lim_i, lim_i:-lim_i, :]  

    #     dist = np.abs(jan - centros[..., None, None, :])  
    #     mask = np.all(dist <= rk, axis=4) 
    #     m = np.count_nonzero(mask, axis=(2, 3))

    #     contagem = np.bincount(m.ravel(), minlength=p.shape[0])
    #     p[:, k] = contagem / ncaixas

    # return p[1:, :]

@njit(parallel=True)
def pmr_plus(img, maxr=41):
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
                        # Checagem Minkowski
                        dist = abs(aux[i,j,0] - aux[x,y,0])
                        if dist <= r[k]:
                            dist = abs(aux[i,j,1] - aux[x,y,1])
                            if dist <= r[k]:
                                dist = abs(aux[i,j,2] - aux[x,y,2])
                                if dist <= r[k]:
                                    m += 1
                                    
                # A Thread anota o 'm' encontrado na sua respectiva linha 'x'
                # ZERO risco de Race Condition!
                p_temp[x, m] += 1
                
        # Quando todas as threads terminarem a imagem inteira para o raio k,
        # nós somamos a coluna inteira do p_temp de forma segura e dividimos:
        for m_idx in range(max_m):
            soma = 0.0
            for x_idx in range(img.shape[0]):
                soma += p_temp[x_idx, m_idx]
            p[m_idx, k] = soma / ncaixas

    return p[1:, :]