import streamlit as st
from streamlit_function import func

func.set_title(__file__)

tab = st.tabs(['데이터 생성', '데이터 전처리', '빅데이터', "웹서버", "스마트팩토리"])

with tab[0]:
    st.header("Data Generation") 
    radio_collection = st.radio(label="generation", label_visibility='collapsed', options=["VAE", "GAN", 'Diffusion Model', "Vision Transformer"], horizontal=True)

    if radio_collection == "VAE":
        st.image("./streamlit_images/computer_skills/autoencoder.png", use_column_width=True)
    elif radio_collection == "GAN":
        st.page_link('./pages/98_Source.py', label='Source', icon="🚨")
        st.image("./streamlit_images/computer_skills/GAN.png", width=1200)
    elif radio_collection == 'Diffusion Model':
        st.text("Stable Diffusion")
        st.markdown("https://huggingface.co/spaces/ehild97/SDTest")
        st.image("./streamlit_images/computer_skills/stable_diffusion.png", use_column_width=True)
    elif radio_collection == "Vision Transformer":
        st.page_link('./pages/98_Source.py', label='Source', icon="🚨")
        st.text('focal transformer')
        st.image("./streamlit_images/computer_skills/focal_transformer.png", use_column_width=True)

with tab[1]:
    st.header("Data Preprocessing")

    radio_processing = st.radio(label="preprocessing", label_visibility='collapsed', options=['Hsv Color Mask', 'Histogram Analysis', "Histogram Backprojection", 
                                                                                 "Labeling", 'Abs Diff', 'Sobel Edge', 'Canny Edge', 'Template Matching', "Albumentations"], horizontal=True)
    if radio_processing == "Hsv Color Mask":
        func.image_resize("hsv_color_mask.png", __file__, 500)

    elif radio_processing == "Histogram Analysis":
        func.image_resize("histogram_analysis.png", __file__, 500)

    elif radio_processing == "Histogram Backprojection":
        func.image_resize("histogram_backprojection_01.png", __file__, 300)
        func.image_resize("histogram_backprojection_02.png", __file__, 300)

    elif radio_processing == "Labeling":
        func.image_resize("labeling.png", __file__, 300)

    elif radio_processing == "Abs Diff":
        func.image_resize("abs_diff.png", __file__, 400)

    elif radio_processing == "Sobel Edge":
        func.image_resize("sobel_edge.png", __file__, 300)

    elif radio_processing == "Canny Edge":
        func.image_resize("canny_edge.png", __file__, 500)

    elif radio_processing == "Template Matching":
        func.image_resize("template_matching.png", __file__, 400)

    elif radio_processing == "Albumentations":
        # st.subheader("Albumentations")
        func.image_resize("albumentations.jpg", __file__, 700)

with tab[2]:
    st.header("BigData")

    radio_bigdata = st.radio(label="bigdata", label_visibility='collapsed', options=["VirtualBox", "Xshell", "Cloudera Manager", 'Hue, Xftp', "Execute Program"], horizontal=True)

    if radio_bigdata == 'VirtualBox':
        func.image_resize('bigdata_05.png' ,__file__, 500)
    elif radio_bigdata == "Xshell":
        func.image_resize('bigdata_04.png' ,__file__, 300)
    elif radio_bigdata == "Cloudera Manager":
        func.image_resize('bigdata_01.png' ,__file__, 600)
    elif radio_bigdata == "Hue, Xftp":
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("./streamlit_images/computer_skills/bigdata_06.png", use_column_width=True)
        with col2:
            st.image("./streamlit_images/computer_skills/bigdata_07.png", use_column_width=True)

    elif radio_bigdata == "Execute Program":
        func.image_resize('bigdata_02.png' ,__file__, 40)
        func.image_resize('bigdata_03.png' ,__file__, 600)
    
with tab[3]:
    st.header("WebServer")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image('./streamlit_images/computer_skills/aws_server.png', use_column_width=True)
    with col2:
        st.image('./streamlit_images/computer_skills/aws_server_web.png', use_column_width=True)
    with col3:
        func.image_resize('tomcat_logo.png', __file__, 150)
        func.image_resize('nginx_logo.png', __file__, 150)

with tab[4]:
    st.header("SmartFactory")    

    st.page_link('./pages/98_Source.py', label='Source', icon="🚨")
    st.image('./streamlit_images/computer_skills/smart_factory.png', use_column_width=True)
