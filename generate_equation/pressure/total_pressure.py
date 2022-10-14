import numpy as np
import os
import os.path
import sys
import random

#Definição dos locais onde os arquivos .tex com as fórmulas em formato Latex serão salvos
tex_path = 'data/'

#Nome para os arquivos que vão seguir o formato torricelli_X.tex, onde X é a ordem de criação deles
tex_name = 'pressure'

if not os.path.exists(tex_path):
    os.makedirs(tex_path)

total_no = 0


#Aqui são definidas as listas com as opções de modificação das fórmulas
total_pressure_list =[r'p_{t}',r'P_{t}',r'p_{T}',r'P_{T}']       
length_total_pressure_list = len(total_pressure_list)

atm_list =[r'p_{atm}',r'P_{atm}']       
length_atm_list = len(atm_list)

g_list = ['g']
length_g_list = len(g_list)

pressure_list = ['p','P']
length_pressure_list = len(pressure_list)

h_list =['h','H']
length_h_list = len(h_list)

#lista de fontes para aumentar a variabilidade das equações
fontlist = ['','lxfonts','arev','anttor','gfsartemisia','fourier']


#inicialização necessária para o arquivo .tex
scale = r'\scalebox{4.0}'
docu_class = '\documentclass[12pt]{article}'
eqnarray = r'\usepackage{eqnarray}' 
asmath =r'\usepackage{amsmath}'
color =r'\usepackage{xcolor}'
font_package = r'\usepackage[T1]{fontenc}'
font_base = r'\usepackage{{{0}}}'
graphics =r'\usepackage{graphicx}'
begin_docu =r'\begin{document}'
end_docu =r'\end{document}'

"""
Aqui inicia o processo de formação das equações, combinando as possibilidades de
variáveis indicadas nas listas acima. Reparar que a quantidade de laços FOR 
é equivalente a quantidade de listas mais 1, que representa a lista de fontes

"""
for font in fontlist:
    for pt in range(length_total_pressure_list):
        for atm in range(length_atm_list):
            for g in range(length_g_list):
                for p in range(length_pressure_list):
                    for h in range(length_h_list):
                        #escrevendo os arquivos .tex
                        total_no = total_no + 1
                        tex_file = os.path.join(tex_path,tex_name)
                        tex_file = tex_file+'_'+str(total_no)+'.tex'
                        f = open(tex_file,"w+")             
                        f.write(docu_class)
                        f.write('\n')
                        f.write(eqnarray)
                        f.write('\n')
                        f.write(asmath)
                        f.write('\n')
                        f.write(graphics)
                        f.write('\n')
                        if font != '':   
                            f.write(font_base.format(font))
                            f.write('\n')
                            f.write(font_package)
                            f.write('\n')
                        f.write(begin_docu)
                        f.write('\n')
                        
                        #nesse ponto a equação em si é escrita conforme sintaxe do LATEX
                        pressure = r'{$'+total_pressure_list[pt]+' = ' + atm_list[atm]+ ' + ' + pressure_list[p]+g_list[g]+h_list[h]+'$}'
                        d = scale + pressure

                        f.write(d)
                        f.write('\n')
                        f.write('\n')
                        f.write('\n')
                        f.write(end_docu)
                        f.close()
                        #print(pressure)