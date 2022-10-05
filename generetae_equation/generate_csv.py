import pandas as pd
import os

"""
Script empregado na geração de arquivo CSV com o caminho e rótulo da imagem a ser classificada. Esse script pega todo o conteúdo dentro de "img_dir", portanto, é possível executar
após a construção das imagens para cada fórumla. O arquivo será salvo no diretório de execução do script.
"""

image_dir = 'crop_image/'

files = []
classes = []
for r,d,f in os.walk(image_dir):
    for file in f:
        if '.png' in file:
            clas = file.split('_')[0]
            files.append(image_dir+file)
            classes.append(clas)

df_classes = pd.DataFrame({'classes':classes,'image_file':files})
df_classes.to_csv('classes.csv',index=False)