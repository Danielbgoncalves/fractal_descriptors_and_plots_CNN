import numpy as np
from numba import njit, prange
from scipy.stats import skew
from joblib import Parallel, delayed
from scipy.ndimage import label

@njit
def _label_4conn(box):
    """
    Retorna num_features (int), tamanho_maior_cluster (int)
    Assim como label do scikit fazia, mas como o numba não converte 
    dessa biblioteca tem q implemnetar NA MÃO
    """
    h, w = box.shape
    labels = -1 * np.ones((h, w), dtype=np.int32)
    stack_i = np.empty(h*w, dtype=np.int32)
    stack_j = np.empty(h*w, dtype=np.int32)
    cur_label = 0
    max_cluster = 0

    for i in range(h):
        for j in range(w):
            if box[i, j] != 0 and labels[i, j] == -1:

                stack_ptr = 0
                stack_i[stack_ptr] = i
                stack_j[stack_ptr] = j
                stack_ptr += 1
                labels[i, j] = cur_label
                comp_size = 0

                while stack_ptr > 0:
                    stack_ptr -= 1
                    ci = stack_i[stack_ptr]
                    cj = stack_j[stack_ptr]
                    comp_size += 1

                    # vizinhos 4 conexos!
                    if ci - 1 >= 0 and box[ci-1, cj] != 0 and labels[ci-1, cj] == -1:
                        labels[ci-1, cj] = cur_label
                        stack_i[stack_ptr] = ci-1
                        stack_j[stack_ptr] = cj
                        stack_ptr += 1
                    if ci + 1 < h and box[ci+1, cj] != 0 and labels[ci+1, cj] == -1:
                        labels[ci+1, cj] = cur_label
                        stack_i[stack_ptr] = ci+1
                        stack_j[stack_ptr] = cj
                        stack_ptr += 1
                    if cj - 1 >= 0 and box[ci, cj-1] != 0 and labels[ci, cj-1] == -1:
                        labels[ci, cj-1] = cur_label
                        stack_i[stack_ptr] = ci
                        stack_j[stack_ptr] = cj-1
                        stack_ptr += 1
                    if cj + 1 < w and box[ci, cj+1] != 0 and labels[ci, cj+1] == -1:
                        labels[ci, cj+1] = cur_label
                        stack_i[stack_ptr] = ci
                        stack_j[stack_ptr] = cj+1
                        stack_ptr += 1

                if comp_size > max_cluster:
                    max_cluster = comp_size
                cur_label += 1

    return cur_label, max_cluster

# ==== PARALELO ====
@njit(parallel=True)
def analisar_um_raio_jit_plus(k, r_k, img_aux):
    """
    Versão em Numba da função que analisa um raio r_k com paralelismo.
    Retorna (k, p_k, g_k, h_k)
    """
    h_img, w_img, _ = img_aux.shape
    ncaixas = (h_img - r_k + 1) * (w_img - r_k + 1)
    if ncaixas <= 0:
        return k, 0.0, 0.0, 0.0

    vetBigClusteres_sum_arr = np.zeros((h_img,), dtype=np.float64)
    ptemp_arr = np.zeros((h_img,), dtype=np.float64)
    gtemp_arr = np.zeros((h_img,), dtype=np.float64)
    
    lim = (r_k / 2.0) - 0.5
    lim_int = int(lim)
    
    # O tamanho da janela é fixo para este raio!
    window_size = int(r_k)

    for x in prange(lim_int, int(h_img - lim)):

        box = np.zeros((window_size, window_size), dtype=np.uint8)
        
        for y in range(lim_int, int(w_img - lim)):
            xi = int(x - lim)
            xf = int(x + lim)
            yi = int(y - lim)
            yf = int(y + lim)

            percCount = 0
            box.fill(0) 

            # Preencher o box local
            for i in range(xi, xf + 1):
                for j in range(yi, yf + 1):
                    # e cor
                    if ( abs(img_aux[i, j, 0] - img_aux[x, y, 0]) <= r_k and
                         abs(img_aux[i, j, 1] - img_aux[x, y, 1]) <= r_k and
                         abs(img_aux[i, j, 2] - img_aux[x, y, 2]) <= r_k ):
                        
                        box[i - xi, j - yi] = 1 
                        percCount += 1

            num_features, tamanho_maior = _label_4conn(box)

            vetBigClusteres_sum_arr[x] += (tamanho_maior / (r_k * r_k))
            ptemp_arr[x] += num_features

            if (percCount / (r_k * r_k)) >= 0.59275:
                gtemp_arr[x] += 1

    # somamos os totais
    p_k = np.sum(ptemp_arr) / ncaixas
    g_k = np.sum(gtemp_arr) / ncaixas
    h_k = np.sum(vetBigClusteres_sum_arr) / ncaixas

    return k, p_k, g_k, h_k

def clustperc_jit_plus(img, maxr):
    aux = img.astype(np.float64)
    r = list(range(3, maxr + 1, 2))

    g = np.zeros(len(r))
    p = np.zeros(len(r))
    h = np.zeros(len(r))

    for k in range(len(r)):
        _, p_k, g_k, h_k = analisar_um_raio_jit_plus(k, r[k], aux)
        p[k] = p_k
        g[k] = g_k
        h[k] = h_k

    AreaCluster = np.trapz(p) # área sobre a curva, uma integral
    AreaPerc = np.trapz(g)
    AreaMaxCluster = np.trapz(h)
    SkewnessCluster = skew(p) # o quão assimétrica é a curva
    SkewnessPerc = skew(g)
    SkewnessMaxCluster = skew(h)
    [MaxCluster,MaxClusterIndex] = np.max(p), np.argmax(p) # o maior valor e seu index (da curva)
    [MaxPerc,MaxPercIndex] = np.max(g), np.argmax(g)
    [MaxMaxCluster,MaxMaxClusterIndex] = np.max(h), np.argmax(h)
    half = int(np.ceil(len(p)/2))
    AreaRatioCluster = np.trapz(p[half:])/np.trapz(p[:half])
    AreaRatioPerc = np.trapz(g[half:])/np.trapz(g[:half])
    AreaRatioMaxCluster = np.trapz(h[half:])/np.trapz(h[:half])

    return {
        'MinkMaxClusterIndex': MaxClusterIndex,
        'MinkMaxPercIndex': MaxPercIndex,
        'MinkMaxMaxClusterIndex': MaxMaxClusterIndex,
        'MinkAreaRatioMaxCluster': AreaRatioMaxCluster,
        'MinkMaxMaxCluster': MaxMaxCluster,
        'MinkSkewnessMaxCluster': SkewnessMaxCluster,
        'MinkAreaMaxCluster': AreaMaxCluster,
        'MinkAreaRatioCluster': AreaRatioCluster,
        'MinkAreaRatioPerc': AreaRatioPerc,
        'MinkMaxCluster': MaxCluster,
        'MinkMaxPerc': MaxPerc,
        'MinkSkewnessCluster': SkewnessCluster, #
        'MinkSkewnessPerc': SkewnessPerc,
        'MinkAreaPerc': AreaPerc,
        'MinkAreaCluster': AreaCluster, #
        'Minkp': p, #
        'Minkg': g, #
        'Minkh': h #
    }

# === SEQUÊNCIAL ====

@njit
def analisar_um_raio_jit(k, r_k, img_aux):
    """
    Versão em Numba da função que analisa um raio r_k.
    Retorna (k, p_k, g_k, h_k)
    """
    h_img, w_img, _ = img_aux.shape
    ncaixas = (h_img - r_k + 1) * (w_img - r_k + 1)
    if ncaixas <= 0:
        return k, 0.0, 0.0, 0.0

    vetBigClusteres_sum = 0.0  
    ptemp = 0.0
    gtemp = 0.0
    lim = (r_k / 2.0) - 0.5

    caixa_idx = 0
    for x in range(int(lim), int(h_img - lim)):
        for y in range(int(lim), int(w_img - lim)):
            xi = int(x - lim)
            xf = int(x + lim)
            yi = int(y - lim)
            yf = int(y + lim)

            rows = xf - xi + 1
            cols = yf - yi + 1
            # criar box local
            box = np.zeros((rows, cols), dtype=np.uint8)

            percCount = 0
            a = 0
            for i in range(xi, xf + 1):
                b = 0
                for j in range(yi, yf + 1):
                    # comparar canais
                    if ( abs(img_aux[i, j, 0] - img_aux[x, y, 0]) <= r_k and
                         abs(img_aux[i, j, 1] - img_aux[x, y, 1]) <= r_k and
                         abs(img_aux[i, j, 2] - img_aux[x, y, 2]) <= r_k ):
                        box[a, b] = 1
                        percCount += 1
                    else:
                        box[a, b] = 0
                    b += 1
                a += 1

            num_features, tamanho_maior = _label_4conn(box)

            vetBigClusteres_sum += (tamanho_maior / (r_k * r_k))
            ptemp += num_features

            if (percCount / (r_k * r_k)) >= 0.59275:
                gtemp += 1

            caixa_idx += 1

    p_k = ptemp / ncaixas
    g_k = gtemp / ncaixas
    h_k = vetBigClusteres_sum / ncaixas

    return k, p_k, g_k, h_k

def clustperc_jit(img, maxr):
    aux = img.astype(np.float64)
    r = list(range(3, maxr + 1, 2))

    g = np.zeros(len(r))
    p = np.zeros(len(r))
    h = np.zeros(len(r))

    for k in range(len(r)):
        _, p_k, g_k, h_k = analisar_um_raio_jit(k, r[k], aux)
        p[k] = p_k
        g[k] = g_k
        h[k] = h_k

    # AreaCluster = np.trapz(p) # área sobre a curva, uma integral
    # AreaPerc = np.trapz(g)
    # AreaMaxCluster = np.trapz(h)
    # SkewnessCluster = skew(p) # o quão assimétrica é a curva
    # SkewnessPerc = skew(g)
    # SkewnessMaxCluster = skew(h)
    # [MaxCluster,MaxClusterIndex] = np.max(p), np.argmax(p) # o maior valor e seu index (da curva)
    # [MaxPerc,MaxPercIndex] = np.max(g), np.argmax(g)
    # [MaxMaxCluster,MaxMaxClusterIndex] = np.max(h), np.argmax(h)
    # half = int(np.ceil(len(p)/2))
    # AreaRatioCluster = np.trapz(p[half:])/np.trapz(p[:half])
    # AreaRatioPerc = np.trapz(g[half:])/np.trapz(g[:half])
    # AreaRatioMaxCluster = np.trapz(h[half:])/np.trapz(h[:half])

    return {
        # 'MinkMaxClusterIndex': MaxClusterIndex,
        # 'MinkMaxPercIndex': MaxPercIndex,
        # 'MinkMaxMaxClusterIndex': MaxMaxClusterIndex,
        # 'MinkAreaRatioMaxCluster': AreaRatioMaxCluster,
        # 'MinkMaxMaxCluster': MaxMaxCluster,
        # 'MinkSkewnessMaxCluster': SkewnessMaxCluster,
        # 'MinkAreaMaxCluster': AreaMaxCluster,
        # 'MinkAreaRatioCluster': AreaRatioCluster,
        # 'MinkAreaRatioPerc': AreaRatioPerc,
        # 'MinkMaxCluster': MaxCluster,
        # 'MinkMaxPerc': MaxPerc,
        # 'MinkSkewnessCluster': SkewnessCluster, #
        # 'MinkSkewnessPerc': SkewnessPerc,
        # 'MinkAreaPerc': AreaPerc,
        # 'MinkAreaCluster': AreaCluster, #
        'Minkp': p, #
        'Minkg': g, #
        'Minkh': h #
    }

# ==== Sem NUMBA ====

def analisar_um_raio(k, r_k, img_aux):
    '''
    Essa função a análise de cluster/percolação para um único raio ed caixa r_k
    '''
    
    ncaixas = (img_aux.shape[0] - r_k + 1) * (img_aux.shape[1] - r_k + 1)
    if ncaixas <= 0: 
        return k, 0.0, 0.0, 0.0

    vetBigClusteres = np.zeros(ncaixas, dtype=np.float64) # armazena, para cada janela, a fração ocupada pelo maior cluster
    ptemp, gtemp = 0, 0 # soma do número de clusters de todas as janelas, quantas janelas satisfazem o crtério de percolação 
    lim = (r_k / 2) - 0.5

    # percorrer os pixels centrais dessas caixas
    caixa_idx = 0
    for x in range(int(lim), int(img_aux.shape[0] - lim)):
        for y in range(int(lim), int(img_aux.shape[1] - lim)):

            # os limites que delimitam a janela de centro x, y
            xi = int( x - lim )
            xf = int( x + lim )
            yi = int( y - lim )
            yf = int( y + lim )
            percCount = 0

            # essa caixa é a janela de centro x, y
            box = np.zeros((xf - xi + 1, yf - yi + 1))
            a = -1
            for i in range(xi, xf + 1):
                a += 1
                b = -1
                for j in range(yi, yf + 1):
                    b += 1
                    if( # Se o pixel i,j for muito parecido com o central box[i,j] recebe valor 1
                        abs(img_aux[i, j, 0] - img_aux[x, y, 0]) <= r_k and
                        abs(img_aux[i, j, 1] - img_aux[x, y, 1]) <= r_k and
                        abs(img_aux[i, j, 2] - img_aux[x, y, 2]) <= r_k
                    ):
                        box[a,b] = 1
                        percCount += 1 # nmr de pixels que diferem do central por <= r_k
                    else:
                        box[a,b] = 0

            structure = np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=np.int8)
            
            labeled, num_features = label(box, structure=structure)

            labels_pos = labeled[labeled > 0]
            if labels_pos.size > 0:
                _, counts = np.unique(labels_pos, return_counts=True)
                tamanho_maior_cluster = np.max(counts)
            else:
                tamanho_maior_cluster = 0

            vetBigClusteres[caixa_idx] = tamanho_maior_cluster/(r_k ** 2)
            ptemp += num_features

            if (percCount / r_k ** 2) >= 0.59275:
                gtemp += 1

            caixa_idx += 1

    p_k = ptemp / ncaixas
    g_k = gtemp / ncaixas
    h_k = np.mean(vetBigClusteres)

    return k, p_k, g_k, h_k

def clustperc(img, maxr):
    aux = img.astype(np.float64)
    r = list(range(3, maxr + 1, 2)) # [3,5,7,...maxr]

    g = np.zeros(len(r))   # valores de percolação para cada tamanho de r
    p = np.zeros(len(r))   # nmr médio de clusters para cada tamanho de r
    h = np.zeros(len(r))   # tamanho médio do maior cluster para cada tamanho de r

    # para cada tamanho caixa *executar em paralelo*
    results = Parallel(n_jobs=-1)(
        delayed(analisar_um_raio)( k, r[k], aux) for k in range(len(r))
    )

    for (k, p_k, g_k, h_k) in results:
        p[int(k)] = p_k
        g[int(k)] = g_k
        h[int(k)] = h_k

    AreaCluster = np.trapz(p) # área sobre a curva, uma integral
    AreaPerc = np.trapz(g)
    AreaMaxCluster = np.trapz(h)
    SkewnessCluster = skew(p) # o quão assimétrica é a curva
    SkewnessPerc = skew(g)
    SkewnessMaxCluster = skew(h)
    [MaxCluster,MaxClusterIndex] = np.max(p), np.argmax(p) # o maior valor e seu index (da curva)
    [MaxPerc,MaxPercIndex] = np.max(g), np.argmax(g)
    [MaxMaxCluster,MaxMaxClusterIndex] = np.max(h), np.argmax(h)
    half = int(np.ceil(len(p)/2))
    AreaRatioCluster = np.trapz(p[half:])/np.trapz(p[:half])
    AreaRatioPerc = np.trapz(g[half:])/np.trapz(g[:half])
    AreaRatioMaxCluster = np.trapz(h[half:])/np.trapz(h[:half])

    return {
        'MinkMaxClusterIndex': MaxClusterIndex,
        'MinkMaxPercIndex': MaxPercIndex,
        'MinkMaxMaxClusterIndex': MaxMaxClusterIndex,
        'MinkAreaRatioMaxCluster': AreaRatioMaxCluster,
        'MinkMaxMaxCluster': MaxMaxCluster,
        'MinkSkewnessMaxCluster': SkewnessMaxCluster,
        'MinkAreaMaxCluster': AreaMaxCluster,
        'MinkAreaRatioCluster': AreaRatioCluster,
        'MinkAreaRatioPerc': AreaRatioPerc,
        'MinkMaxCluster': MaxCluster,
        'MinkMaxPerc': MaxPerc,
        'MinkSkewnessCluster': SkewnessCluster, #
        'MinkSkewnessPerc': SkewnessPerc,
        'MinkAreaPerc': AreaPerc,
        'MinkAreaCluster': AreaCluster, #
        'Minkp': p, #
        'Minkg': g, #
        'Minkh': h #
    }