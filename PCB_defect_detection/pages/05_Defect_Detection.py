import streamlit as st
from streamlit_function import func

func.set_title(__file__)

with st.sidebar:
    radio_sidebar = st.radio(label="", label_visibility='collapsed', options=["흑백 데이터셋", "실물 데이터셋"])
    