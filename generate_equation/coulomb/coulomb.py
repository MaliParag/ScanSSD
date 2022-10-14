import numpy as np
import os
import os.path
import sys
import random

#Definição dos locais onde os arquivos .tex com as fórmulas em formato Latex serão salvos
tex_path = 'data/'

#Nome para os arquivos que vão seguir o formato torricelli_X.tex, onde X é a ordem de criação deles
tex_name = 'coulomb'

if not os.path.exists(tex_path):
    os.makedirs(tex_path)

total_no = 0


#Aqui são definidas as listas com as opções de modificação das fórmulas
force_list =['F',r'F_{c}']
length_force_list = len(force_list)

k_list =['k','K']       
length_k_list = len(k_list)

q_list =['q','Q']       
length_q_list = len(q_list)

r_list = ['r','R','d','D']
length_r_list = len(r_list)

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
    for force in range(length_force_list):
        for k in range(length_k_list):
            for q in range(length_q_list):
                for radios in range(length_r_list):
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
                    #print(f)
                    #print(force_list[f])
                    coulomb = r'{$'+force_list[force]+' = '+k_list[k] + r'\frac{'+ q_list[q]+'_1'+q_list[q]+'_2}{'+r_list[radios]+'^2}$}'
                    d = scale + coulomb

                    f.write(d)
                    f.write('\n')
                    f.write('\n')
                    f.write('\n')
                    f.write(end_docu)
                    f.close()
                    #print(coulomb)