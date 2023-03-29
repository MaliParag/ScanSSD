from api import scanssd
import streamlit as st
import os
import base64
import time
import glob
import uuid

path_list = ['images','pdf','logs','results','crop','annotations']
root_folder = '../arquivos'
uuid_folder = None
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

#criando estrutura mínima de diretórios
@st.cache_data
def create_folders(root_folder,path_list):
    #uuid_folder = (str(uuid.uuid4())).split('-',1)[0]
    uuid_folder = 'test'
    os.mkdir(os.path.join(root_folder,uuid_folder))
    paths = [os.path.join(root_folder,uuid_folder,path) for path in path_list]
    for path in paths:
        os.mkdir(path)
    return uuid_folder

uuid_folder = create_folders(root_folder=root_folder,path_list=path_list)

file = st.file_uploader("Please choose a file", type="pdf")

if file is not None:
    # file_details = {"FileName": file.name, "FileType": file.type}
    # st.write(file_details)
    st.error(f"Do you really want to process the file: {file.name}")
    if st.button("Yes"):
        st.session_state["process"] = True
        file_path = os.path.join(root_folder,uuid_folder,'pdf', file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        show_pdf(file_path)

        st.success("File Uploaded")
        with st.spinner('Wait for it... processing...'):

            annot_images = scanssd.test_gtdb(trained_model='weights\AMATH512_e1GTDB.pth', visual_threshold=0.6, cuda=False,
                              verbose=False, exp_name=uuid_folder, model_type=512,
                              use_char_info=False, limit=-1, cfg="hboxes512", batch_size=16, num_workers=4,
                              kernel=[1, 5], padding=[3, 3], neg_mining=True,
                              stride=0.1, window=1200, root_folder=root_folder)
            
        #annot_images = os.path.join(root_folder,uuid_folder,'annotated')
        st.success("Sucess!")

        # read image from response
        file_name = (file.name).split('.',1)[0]
        print(os.path.join(annot_images,file_name,'*.png'))
        images_to_plot = glob.glob(os.path.join(annot_images,file_name,'*.png'))
        st.write('images read')

        # plot images in 3 columns
        grid_size = 3
        image_groups = []
        for i in range(0, len(images_to_plot), grid_size):
            image_groups.append(images_to_plot[i:i+grid_size])

        for image_group in image_groups:
            streamlit_columns = st.columns(grid_size)
            for i, image in enumerate(image_group):
                streamlit_columns[i].image(image)
        
        ## Aqui pode exibir o latex gerado a partir do arquivo
        ## root_folder/uuid_folder/latex.txt
        ## depois para cada linha desse arquivo, fazer uma chamada
        ## para a API do chatGPT e exibir o resultado
