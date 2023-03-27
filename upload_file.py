from api import scanssd
import streamlit as st
import os
import base64
import time
import glob


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

file = st.file_uploader("Please choose a file", type="pdf")

if file is not None:
    # file_details = {"FileName": file.name, "FileType": file.type}
    # st.write(file_details)
    st.error(f"Do you really want to process the file: {file.name}")
    if st.button("Yes"):
        st.session_state["process"] = True
        file_path = os.path.join("./uploded", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        # show_pdf(file_path)

        st.success("File Uploaded")
        with st.spinner('Wait for it... processing...'):

            scanssd.test_gtdb(trained_model='weights\AMATH512_e1GTDB.pth', visual_threshold=0.6, cuda=False,
                              verbose=False, exp_name='teste_api', model_type=512,
                              use_char_info=False, limit=-1, cfg="hboxes512", batch_size=16, num_workers=4,
                              kernel=[1, 5], padding=[3, 3], neg_mining=True,
                              stride=0.1, window=1200, root_folder='../arquivos')

        st.success("Sucess!")

        # read image from response
        images_to_plot = glob.glob("./resources/mock_results/*.jpeg")
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
