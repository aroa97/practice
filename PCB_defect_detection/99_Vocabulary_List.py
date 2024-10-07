import streamlit as st
from streamlit_function import func

func.set_title(__file__)

col1, col2, col3 = st.columns(3)

with col1 :
    st.markdown("**OKR**")
    st.text("Objectives and Key Results (목표와 핵심 결과)")

    st.markdown("**VAE**")
    st.text("Variational Autoencoder (변이형 오토인코터)")

    st.markdown('**GAN**')
    st.text("Generative Adversarial Networks (적대적 생성 신경망)")