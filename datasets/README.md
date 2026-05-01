Esse diretório foi criado com o arquivo main_pipeline.py. Ele utiliza dois datasets, um de Displasia e outro de Cancer de Pulmão que não estão carregados nesse repositorio. 

O arquivo main_pipeline.py é o script principal que executa o pipeline completo de extração de features e geração de imagens a partir dos datasets. Ele depende dos arquivos de extração dos descritores fractais na pasta features_extraction/, e utiliza os algoritmos na pasta feature_to_image/ para transformar os descritores em imagens. Tudo isso de acordo com a referência em MATLAB do Guilherme Freire Roberto.

As imagens geradas pelo main_pipeline.py não foram carregadas nesse repositório, mas o script pode ser executado localmente para gerar as imagens a partir dos datasets de Displasia e Cancer de Pulmão ou a partir desses CSVs. 
