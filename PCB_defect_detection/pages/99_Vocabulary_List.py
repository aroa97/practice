import streamlit as st
from streamlit_function import func

func.set_title(__file__)

col1, col2, col3 = st.columns(3)

with col1 :
    st.markdown("**VAE**")
    st.text("Variational Autoencoder (변이형 오토인코터)")