import pandas as pd
import numpy as np

def reorganizar_e_expandir_df(df):
    
    # Nova ordem das colunas
    nova_ordem = [
        'Minkp', 'Minkg', 'Minkh', 'MinkLAC', 'Minknn', 
        'Euclp', 'Euclg', 'Euclh', 'EuclLAC', 'Euclnn', 
        'Manhp', 'Manhg', 'Manhh', 'ManhLAC', 'Manhnn', 

        'MinkAreaCluster', 'MinkSkewnessCluster', 'MinkAreaRatioCluster', 'MinkMaxCluster', 'MinkMaxClusterIndex',
        'MinkAreaPerc', 'MinkSkewnessPerc', 'MinkAreaRatioPerc', 'MinkMaxPerc', 'MinkMaxPercIndex', 
        'MinkAreaMaxCluster', 'MinkSkewnessMaxCluster', 'MinkAreaRatioMaxCluster', 'MinkMaxMaxCluster', 'MinkMaxMaxClusterIndex', 

        'EuclAreaCluster', 'EuclSkewnessCluster', 'EuclAreaRatioCluster', 'EuclMaxCluster', 'EuclMaxClusterIndex',
        'EuclAreaPerc', 'EuclSkewnessPerc', 'EuclAreaRatioPerc', 'EuclMaxPerc', 'EuclMaxPercIndex', 
        'EuclAreaMaxCluster', 'EuclSkewnessMaxCluster', 'EuclAreaRatioMaxCluster', 'EuclMaxMaxCluster', 'EuclMaxMaxClusterIndex',

        'ManhAreaCluster', 'ManhSkewnessCluster', 'ManhAreaRatioCluster', 'ManhMaxCluster', 'ManhMaxClusterIndex',
        'ManhAreaPerc', 'ManhSkewnessPerc', 'ManhAreaRatioPerc', 'ManhMaxPerc', 'ManhMaxPercIndex', 
        'ManhAreaMaxCluster', 'ManhSkewnessMaxCluster', 'ManhAreaRatioMaxCluster', 'ManhMaxMaxCluster', 'ManhMaxMaxClusterIndex',

        'MinkAreaLAC', 'MinkSkewnessLAC', 'MinkAreaRatioLAC', 'MinkMaxLAC', 'MinkMaxLACIndex',
        'EuclAreaLAC', 'EuclSkewnessLAC', 'EuclAreaRatioLAC', 'EuclMaxLAC', 'EuclMaxLACIndex', 
        'ManhAreaLAC', 'ManhSkewnessLAC', 'ManhAreaRatioLAC', 'ManhMaxLAC', 'ManhMaxLACIndex',

        'MinkDF', 'EuclDF', 'ManhDF'
    ]
    
    # Identificar colunas que terminam com 'p', 'g', 'h', 'LACs', 'nn' (vetores a expandir)
    sufixos_vetores = ['p', 'g', 'h', 'MinkLAC', 'EuclLAC', 'ManhLAC', 'nn']
    colunas_vetores = [col for col in nova_ordem if any(col.endswith(suf) for suf in sufixos_vetores)]
    
    ordem_final = []
    dados_expandidos = {}
    
    for col in nova_ordem:
        if col in colunas_vetores and col in df.columns:

            primeiro_array = df[col].iloc[0]
            if isinstance(primeiro_array, (np.ndarray, list)):
                n_elementos = len(primeiro_array)
            else:
                n_elementos = 20  # valor padrão
            
            for i in range(n_elementos):
                nome_nova_col = f"{col}{i+1}"
                ordem_final.append(nome_nova_col)

                dados_expandidos[nome_nova_col] = [
                    arr[i] if isinstance(arr, (np.ndarray, list)) and len(arr) > i else np.nan 
                    for arr in df[col]
                ]
        elif col in df.columns:

            ordem_final.append(col)
            dados_expandidos[col] = df[col].values
    
    df_novo = pd.DataFrame(dados_expandidos)
    df_novo = df_novo[ordem_final]
    
    return df_novo

def reorganizar_e_expandir_csv2(df):

    
    # Nova ordem das colunas
    nova_ordem = [
        'Minkp', 'Minkg', 'Minkh', 'MinkLAC', 'Minknn', 
        'Euclp', 'Euclg', 'Euclh', 'EuclLAC', 'Euclnn', 
        'Manhp', 'Manhg', 'Manhh', 'ManhLAC', 'Manhnn', 

        'MinkAreaCluster', 'MinkSkewnessCluster', 'MinkAreaRatioCluster', 'MinkMaxCluster', 'MinkMaxClusterIndex',
        'MinkAreaPerc', 'MinkSkewnessPerc', 'MinkAreaRatioPerc', 'MinkMaxPerc', 'MinkMaxPercIndex', 
        'MinkAreaMaxCluster', 'MinkSkewnessMaxCluster', 'MinkAreaRatioMaxCluster', 'MinkMaxMaxCluster', 'MinkMaxMaxClusterIndex', 
        
        'EuclAreaCluster', 'EuclSkewnessCluster', 'EuclAreaRatioCluster', 'EuclMaxCluster', 'EuclMaxClusterIndex',
        'EuclAreaPerc', 'EuclSkewnessPerc', 'EuclAreaRatioPerc', 'EuclMaxPerc', 'EuclMaxPercIndex', 
        'EuclAreaMaxCluster', 'EuclSkewnessMaxCluster', 'EuclAreaRatioMaxCluster', 'EuclMaxMaxCluster', 'EuclMaxMaxClusterIndex',
        
        'ManhAreaCluster', 'ManhSkewnessCluster', 'ManhAreaRatioCluster', 'ManhMaxCluster', 'ManhMaxClusterIndex',
        'ManhAreaPerc', 'ManhSkewnessPerc', 'ManhAreaRatioPerc', 'ManhMaxPerc', 'ManhMaxPercIndex', 
        'ManhAreaMaxCluster', 'ManhSkewnessMaxCluster', 'ManhAreaRatioMaxCluster', 'ManhMaxMaxCluster', 'ManhMaxMaxClusterIndex'

        'MinkAreaLAC', 'MinkSkewnessLAC', 'MinkAreaRatioLAC', 'MinkMaxLAC', 'MinkMaxLACIndex',
        'EuclAreaLAC', 'EuclSkewnessLAC', 'EuclAreaRatioLAC', 'EuclMaxLAC', 'EuclMaxLACIndex', 
        'ManhAreaLAC', 'ManhSkewnessLAC', 'ManhAreaRatioLAC', 'ManhMaxLAC', 'ManhMaxLACIndex',

        'MinkDF', 'EuclDF', 'ManhDF'
    ]
    
    # Identificar colunas que terminam com 'p', 'g' ou 'h' (vetores a expandir)
    sufixos_vetores = ['p', 'g', 'h', 'MinkLAC', 'EuclLAC', 'ManhLAC', 'nn']
    
    # Função para converter string de array para numpy array
    def parse_array(val):
        if pd.isna(val):
            return None
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, str):
            # Remove colchetes e quebras de linha, depois converte
            val_clean = val.replace('[', '').replace(']', '').replace('\n', ' ')
            # Split por espaços e converte para float
            try:
                return np.array([float(x) for x in val_clean.split() if x])
            except:
                return None
        return None
    
    # Lista para armazenar os nomes das colunas na ordem final
    ordem_final = []
    
    # Dicionário para armazenar os dados
    dados_expandidos = {}
    
    # Processar cada coluna na ordem definida
    for col in nova_ordem:
        if not col in df.columns:
            # Pular colunas que não existem no DataFrame
            continue
            
        # Verificar se é uma coluna de vetor (termina com p, g ou h)
        if col[-1] in sufixos_vetores:
            # Converter strings para arrays se necessário
            arrays = df[col].apply(parse_array)
            
            # Determinar número de elementos
            primeiro_array = None
            for arr in arrays:
                if arr is not None:
                    primeiro_array = arr
                    break
            
            if primeiro_array is not None:
                n_elementos = len(primeiro_array)
                
                # Criar colunas individuais para cada elemento do array
                for i in range(n_elementos):
                    nome_nova_col = f"{col}{i+1}"
                    ordem_final.append(nome_nova_col)
                    # Extrair o elemento i de cada array
                    dados_expandidos[nome_nova_col] = [
                        arr[i] if arr is not None and len(arr) > i else np.nan 
                        for arr in arrays
                    ]
            else:
                # Se não conseguiu parse, adiciona coluna original
                ordem_final.append(col)
                dados_expandidos[col] = df[col].values
        else:
            # Manter coluna como está (não é vetor)
            ordem_final.append(col)
            dados_expandidos[col] = df[col].values
    
    # Criar novo DataFrame com as colunas na ordem correta
    df_novo = pd.DataFrame(dados_expandidos)
    
    return df_novo

def gerar_df_exemplo(n_rows=5, seed=42):
    
    np.random.seed(seed)
    data = {}
    
    colunas_vetores = ['Minkp', 'Minkg', 'Minkh', 'MinkLAC', 'Minknn', 'Euclp', 'Euclg', 'Euclh', 'EuclLAC', 'Euclnn', 'Manhp', 'Manhg', 'Manhh', 'ManhLAC', 'Manhnn']
    
    for col in colunas_vetores:
        data[col] = [np.random.rand(20) for _ in range(n_rows)]
    
    colunas_escalares = [
        'MinkAreaCluster', 'MinkSkewnessCluster', 'MinkAreaRatioCluster', 'MinkMaxCluster', 'MinkMaxClusterIndex',
        'MinkAreaPerc', 'MinkSkewnessPerc', 'MinkAreaRatioPerc', 'MinkMaxPerc', 'MinkMaxPercIndex', 
        'MinkAreaMaxCluster', 'MinkSkewnessMaxCluster', 'MinkAreaRatioMaxCluster', 'MinkMaxMaxCluster', 'MinkMaxMaxClusterIndex', 
        'EuclAreaCluster', 'EuclSkewnessCluster', 'EuclAreaRatioCluster', 'EuclMaxCluster', 'EuclMaxClusterIndex',
        'EuclAreaPerc', 'EuclSkewnessPerc', 'EuclAreaRatioPerc', 'EuclMaxPerc', 'EuclMaxPercIndex', 
        'EuclAreaMaxCluster', 'EuclSkewnessMaxCluster', 'EuclAreaRatioMaxCluster', 'EuclMaxMaxCluster', 'EuclMaxMaxClusterIndex',
        'ManhAreaCluster', 'ManhSkewnessCluster', 'ManhAreaRatioCluster', 'ManhMaxCluster', 'ManhMaxClusterIndex',
        'ManhAreaPerc', 'ManhSkewnessPerc', 'ManhAreaRatioPerc', 'ManhMaxPerc', 'ManhMaxPercIndex', 
        'ManhAreaMaxCluster', 'ManhSkewnessMaxCluster', 'ManhAreaRatioMaxCluster', 'ManhMaxMaxCluster', 'ManhMaxMaxClusterIndex'
    ]
    
    for col in colunas_escalares:
        if 'Index' in col:
            data[col] = np.random.randint(0, 20, n_rows).astype(float)
        else:
            data[col] = np.random.rand(n_rows) * 10
    
    df_exemplo = pd.DataFrame(data)
    
    return df_exemplo

def printa():
    print('import certo')

def reorganizar_e_expandir_df_benchmark(df):
    
    # Nova ordem das colunas
    nova_ordem = [
        'img_id',
        'Minkp', 'Minkg', 'Minkh', 
        'Euclp', 'Euclg', 'Euclh',  
        'Manhp', 'Manhg', 'Manhh',  
        'LAC', 'nn', 'DF'
    ]
    
    # Identificar colunas que terminam com 'p', 'g', 'h', 'LAC', 'nn' (vetores a expandir)
    sufixos_vetores = ['p', 'g', 'h', 'LAC', 'nn']
    colunas_vetores = [col for col in nova_ordem if any(col.endswith(suf) for suf in sufixos_vetores)]
    
    ordem_final = []
    dados_expandidos = {}
    
    for col in nova_ordem:
        if col in colunas_vetores and col in df.columns:

            primeiro_array = df[col].iloc[0]
            if isinstance(primeiro_array, (np.ndarray, list)):
                n_elementos = len(primeiro_array)
            else:
                n_elementos = 20  # valor padrão
            
            for i in range(n_elementos):
                nome_nova_col = f"{col}{i+1}"
                ordem_final.append(nome_nova_col)

                dados_expandidos[nome_nova_col] = [
                    arr[i] if isinstance(arr, (np.ndarray, list)) and len(arr) > i else np.nan 
                    for arr in df[col]
                ]
        elif col in df.columns:

            ordem_final.append(col)
            dados_expandidos[col] = df[col].values
    
    df_novo = pd.DataFrame(dados_expandidos)
    df_novo = df_novo[ordem_final]
    
    return df_novo