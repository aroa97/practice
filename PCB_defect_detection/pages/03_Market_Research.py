import streamlit as st
from streamlit_function import func

func.set_title(__file__)

tab = st.tabs(['시장조사','OKR'])

with tab[0]:
    st.text('')
with tab[-1]:
    st.subheader('Objectives and Key Results')