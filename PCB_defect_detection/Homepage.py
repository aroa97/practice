import streamlit as st

if __name__ == "__main__":
    st.set_page_config(page_title="Homepage", layout="wide")

    st.title("PCB Defect Detection Project")

    st.image("./streamlit_images/background_image2.png", width=500)
    
    st.markdown("PCB 불량 검출 프로젝트는 인공지능 기반의 머신러닝 및 딥러닝 알고리즘을 통해 인쇄 회로 기판(PCB)의 결함을 자동으로 식별하고")
    st.markdown("실시간으로 PCB의 품질을 평가해 결함을 조기에 발견하여 생산 효율성을 높이는 데 기여하기 위한 프로젝트입니다.")