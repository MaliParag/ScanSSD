import os
import os.path
import PyPDF2
from wand.image import Image
import io
import glob
import cv2
import numpy as np
from PIL import Image as I

"""
Script empregado na geração das fórmulas a partir do arquivo .tex gerado com o script próprio
para cada fórmula, portanto, a partir desse ponto a execução é igual para todas as fórmulas.
"""


latex_path = 'torricelli/data/' #define o local onde os .tex foram salvos
pdf_path = 'torricelli/data/' #onde vai salvar o PDF, intermediário no processo de criação da imagem
image_path = 'torricelli/data/' #onde vai salvar o .PNG gerado para a página do PDF, também intermediário à criação da imagem final
crop_image_path = 'crop_image/' #onde salva a imagem já cortada no tamanho que considera apenas a fórmula e não a página toda

#parâmetros para o crop da imagem
extra_part = 40
half_part = 20

del_files = True #Escolhe se os arquivos intermediários, PDF e Imagem da página completa, devem ser apagados após a construção da imagem cortada.

a = 'pdflatex -quiet ' #comando para chamar a aplicação pdflatex
a1 = "del " #para linux usar rm

def generate_pdf(latex_path,a,a1):
    base_path = os.getcwd()
    os.chdir(base_path + '/'+latex_path)
    for file_name in os.listdir():
        if file_name.endswith('.tex'):
            f1 = file_name.split('.')
            f2 = f1[0]
            f3 = f1[1]
            if f3 == "tex":
                print("file_name", file_name)
                b = file_name
                c = a+b
                os.system(c)
                # remove .ps, .aux, .log, .dvi files 

                f11 = f1[0]+".ps"
                f12 = f1[0]+".aux"
                f13 = f1[0]+".log"
                f14 = f1[0]+".dvi"
                
                fa1 = a1+f11
                fa2 = a1+f12
                fa3 = a1+f13
                fa4 = a1+f14
                
                os.system(fa1)
                os.system(fa2)
                os.system(fa3)
                os.system(fa4)
    os.chdir(base_path)

print("Gerando PDF a partir do LATEX!!")
generate_pdf(latex_path,a,a1)

#---------------------------------------------------------------------------------------
#generate image from pdf

def pdf_page_to_png(src_pdf, pagenum = 0, resolution = 100):
    dst_pdf = PyPDF2.PdfFileWriter()
    dst_pdf.addPage(src_pdf.getPage(pagenum))
    pdf_bytes = io.BytesIO()
    dst_pdf.write(pdf_bytes)
    pdf_bytes.seek(0)
    img = Image(file = pdf_bytes, resolution = resolution)
    return img

def generate_image(pdf_path,image_path):
    for filename in os.listdir(pdf_path):
        if filename.endswith('.pdf'):
            print('filename',filename)
            r =filename.replace(".pdf","")
            
            full_name = os.path.join(pdf_path,filename)
            src_pdf = PyPDF2.PdfFileReader(full_name, "rb")
            no_page = src_pdf.getNumPages()
            for j in range (0, 1):
                img = pdf_page_to_png(src_pdf, pagenum = j, resolution = 100)
                file_name = r+".png"
                b = os.path.join(image_path,file_name)
                img.save(filename=b)
                img = cv2.imread(b)
                img_height, img_width, img_ch = img.shape
                act_image = np.zeros((img_height, img_width, 3), np.uint8)
                act_image = img
                act_image1 = I.fromarray(act_image)
                act_image1.save(b,'png')

print("Gerando Imagens a partir do PDF")
generate_image(pdf_path,image_path)

#---------------------------------------------------------------
#Generate cropped image
from PIL import ImageFilter, Image

if not os.path.exists(crop_image_path):
    os.makedirs(crop_image_path)

def crop_image(image_path,crop_image_path):
    for filename in os.listdir(image_path):
        if filename.endswith('.png'): 
            print(filename)
            #read each images from folder
            img = cv2.imread(os.path.join(image_path,filename)) 
            height, width, chal = img.shape
            for i in range(400):
                flag =0
                for j in range (width-1):
                    if img[i,j,0]==255 and img[i,j+1,0]==0:
                        flag = 1
                        col_index1 =j
                        break
                if flag ==1:
                    row_index1 = i
                    break
            #print('height, width', row_index1, col_index1)
            
            # find bottom point
            for i in range(400, 100, -1):
                flag =0
                for j in range (width-1):
                    if img[i,j,0]==255 and img[i,j+1,0]==0:
                        flag = 1
                        col_index2 =j
                        break
                if flag ==1:
                    row_index2 = i
                    break
            #print('height, width', row_index2, col_index2)

            #find left point
            for j in range(width-1):
                flag =0
                for i in range (400):
                    if img[i,j,0]==255 and img[i+1,j,0]==0:
                        flag = 1
                        row_index3 =i
                        break
                if flag ==1:
                    col_index3 = j
                    break
            #print('height, width', row_index3, col_index3)

            #find right point
            for j in range(width-1, 0, -1):
                flag =0
                for i in range (400):
                    if img[i,j,0]==255 and img[i+1,j,0]==0:
                        flag = 1
                        row_index4 =i
                        break
                if flag ==1:
                    col_index4 = j
                    break
            #print('height, width', row_index4, col_index4)
            
            #find four corner points of crop equation
            #find (x1, y1)
            if col_index3<col_index1:
                y1 = col_index3
            else:
                y1 = col_index1    

            if row_index3<row_index1:
                x1 = row_index3
            else:
                x1 = row_index1     

            #find(x2,y2)
            if col_index1<col_index4:
                y2 = col_index4
            else:
                y2 = col_index1     
            
            if row_index1<row_index4:
                x2 = row_index1
            else:
                x2 = row_index4

            #fin(x3, y3) 
            if col_index2<col_index4:
                y3 = col_index4
            else:
                y3 = col_index2

            if row_index2<row_index4:
                x3 = row_index4
            else:
                x3 = row_index2

            #find(x4, y4) 
            if col_index3<col_index2:
                y4 =col_index3
            else:
                y4 = col_index2

            if row_index3<row_index2:
                x4 = row_index2
            else:
                x4 = row_index3

            m = 850-y2-1 
            crop_height = abs(x1-x4)+10
            if m>5:
                crop_width = abs(y1-y2)+5
            else:
                crop_width = abs(y1-y2)

            crop_image = np.zeros((crop_height+extra_part, crop_width+extra_part, 3), np.uint8) 

            for i in range (crop_height+extra_part):
                for j in range(crop_width+extra_part):
                    if crop_image[i,j,0]==0 and crop_image[i,j,1]==0 and crop_image[i,j,2]==0 :
                        crop_image[i,j,0] =255
                        crop_image[i,j,1] =255
                        crop_image[i,j,2] =255

            for i in range (crop_height):
                for j in range (crop_width):
                    crop_image[i+half_part, j+half_part, 0] = img[i+x1, j+y1, 0]
                    crop_image[i+half_part, j+half_part, 1] = img[i+x1, j+y1, 1]
                    crop_image[i+half_part, j+half_part, 2] = img[i+x1, j+y1, 2]            

            crop_image1 = Image.fromarray(crop_image)
            crop_image1.save(os.path.join(crop_image_path,filename),'PNG')

print("Recortando Imagens")
crop_image(image_path,crop_image_path)

if del_files:
    os.chdir(pdf_path)
    os.system("del *.pdf")
    os.system("del *.png")

