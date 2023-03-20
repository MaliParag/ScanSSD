from PIL import Image
import os
import shutil
import numpy as np

# Função para Crop e Resize de Imagens das fórmulas detectadas

def crop_resize(input_dir,output_dir,bouding_dir,file_name,width=100,quality=60):
    
    list_files = os.listdir(bouding_dir)
    bouding_file = [file for file in list_files if '.math' in file]
    bouding_file = os.path.join(bouding_dir,bouding_file[0])
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    if os.path.exists(bouding_file):
        data = np.genfromtxt(bouding_file, delimiter=',')
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        print(data)
    i = 0
    for pg,x1,y1,x2,y2 in data:
        page = int(pg)
        img = Image.open(os.path.join(input_dir,'images',file_name,str(page+1)+'.png'))
        cropped = img.crop(box=(x1-30,y1-30,x2+30,y2+30))
        cropped = resize_image(cropped,width)
        cropped.save(os.path.join(output_dir,str(page+1)+'_'+str(i)+'.png'),optimize=True, quality=quality)
        i = i+1
        print(pg)

def resize_image(image,width):
  base_width = width
  width_percent = (base_width / float(image.size[0]))
  hsize = int((float(image.size[1]) * float(width_percent)))
  image = image.resize((base_width, hsize), Image.ANTIALIAS)
  return image