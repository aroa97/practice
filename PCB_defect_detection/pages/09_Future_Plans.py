import streamlit as st
from streamlit_function import func

func.set_title(__file__)

st.subheader("VAE")
# st.text("Variational Autoencoder")
st.image("./streamlit_images/data_generation/autoencoder.png", use_column_width=True)