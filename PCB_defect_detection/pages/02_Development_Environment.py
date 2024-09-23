import streamlit as st
from streamlit_function import func

func.set_title(__file__)

# 사용한 프로그래밍 언어 : Python, JupyterNotebook
# 가상 환경 관리 : anaconda3
# 데이터 분석 : Pandas, Numpy
# 데이터 시각화 : Matplotlib, Seaborn
# 데이터베이스 : MySQL, Oracle
# 머신 러닝 및 딥 러닝 : scikit-learn, TensorFlow(Keras)
# 웹 App : Streamlit(Github 연동)

tab = st.tabs(["사용한 프로그래밍 언어", "가상 환경 관리", "데이터 분석", "데이터 시각화", "데이터베이스", "머신 러닝 및 러닝", "웹 App"])

with tab[0]:
    st.subheader("사용한 프로그래밍 언어")
    st.text("Python")

    path = "./streamlit_images/logo/python_logo.png"
    func.image_resize(path, 200)
    
with tab[1]:
    st.subheader("가상 환경 관리")
    st.text("Anaconda")

    path = "./streamlit_images/logo/anaconda_logo.png"
    func.image_resize(path, 200)

with tab[2]:
    st.subheader("데이터 분석")
    st.text("OpenCV, Numpy, Pandas")
    col1, col2, col3, _ = st.columns(4)
    with col1 :
        path = "./streamlit_images/logo/opencv_logo.png"
        func.image_resize(path, 250)
    with col3 :
        path = "./streamlit_images/logo/pandas_logo.jpg"
        func.image_resize(path, 200)
    with col2 :
        path = "./streamlit_images/logo/numpy_logo.png"
        func.image_resize(path, 200)
    

with tab[3]:
    st.subheader("데이터 시각화")
    st.text("Matplotlib, Seaborn, Mlxtend")
    col1, col2, col3, _ = st.columns(4)
    with col1 :
        path = "./streamlit_images/logo/matplotlib_logo.png"
        func.image_resize(path, 200)
    with col2 :
        path = "./streamlit_images/logo/seaborn_logo.png"
        func.image_resize(path, 200)
    with col3 :
        path = "./streamlit_images/logo/mlxtend_logo.png"
        func.image_resize(path, 200)

with tab[4]:
    st.subheader("데이터베이스")
    st.markdown("MySQL, Oracle")

    col1, col2, _, _ = st.columns(4)
    with col1 :
        path = "./streamlit_images/logo/mysql_logo.png"
        func.image_resize(path, 200)
    with col2 :
        path = "./streamlit_images/logo/oracle_logo.png"
        func.image_resize(path, 200)

with tab[5]:
    st.subheader("머신 러닝 및 딥 러닝")
    st.markdown("Scikit-learn, TensorFlow")
    col1, col2, _ = st.columns(3)
    with col1 :
        path = "./streamlit_images/logo/scikit_learn_logo.png"
        func.image_resize(path, 200)
    with col2 :
        path = "./streamlit_images/logo/tensorflow_logo.png"
        func.image_resize(path, 200)

with tab[6]:
    st.subheader("웹 App")
    st.markdown("Streamlit, Jupyter")

    col1, col2, _ = st.columns(3)
    with col1:
        path = "./streamlit_images/logo/streamlit_logo.png"
        func.image_resize(path, 200)
    with col2:
        path = "./streamlit_images/logo/jupyter_logo.png"
        func.image_resize(path, 200)
