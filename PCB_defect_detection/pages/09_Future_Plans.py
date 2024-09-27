import streamlit as st
from streamlit_function import func

func.set_title(__file__)

tab = st.tabs(['데이터 전처리', '빅데이터'])

with tab[0]:
    st.header("Data Preprocessing")

    radio_processing = st.radio(label="", label_visibility='collapsed', options=["VAE", "Albumentations"], horizontal=True)
    if radio_processing == "VAE":
        # st.subheader("VAE")
        # st.text("Variational Autoencoder")
        st.image("./streamlit_images/future_plans/autoencoder.png", use_column_width=True)
    elif radio_processing == "Albumentations":
        # st.subheader("Albumentations")
        func.image_resize("albumentations.jpg", __file__, 700)

with tab[1]:
    st.header("BigData")

    radio_bigdata = st.radio(label="", label_visibility='collapsed', options=["VirtualBox", "Xshell", "Cloudera Manager", 'Hue, Xftp', "Execute Program"], horizontal=True)

    if radio_bigdata == 'VirtualBox':
        func.image_resize('bigdata_05.png' ,__file__, 500)
    elif radio_bigdata == "Xshell":
        func.image_resize('bigdata_04.png' ,__file__, 300)
    elif radio_bigdata == "Cloudera Manager":
        func.image_resize('bigdata_01.png' ,__file__, 600)
    elif radio_bigdata == "Hue, Xftp":
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("./streamlit_images/future_plans/bigdata_06.png", use_column_width=True)
        with col2:
            st.image("./streamlit_images/future_plans/bigdata_07.png", use_column_width=True)

    elif radio_bigdata == "Execute Program":
        func.image_resize('bigdata_02.png' ,__file__, 40)
        func.image_resize('bigdata_03.png' ,__file__, 600)
    
    
    
