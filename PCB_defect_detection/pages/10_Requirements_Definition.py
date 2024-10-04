import streamlit as st
from streamlit_function import func

func.set_title(__file__)

tab = st.tabs(['요구공학', '기능 요구사항', '비기능 요구사항'])

with tab[0]:
    st.subheader('Requirements Engineering')
    st.text("요구공학은 시스템 개발에 필요한 요구사항을 도출하여 문서화(명세화)하고 이를 체계적 으로 검증 및 관리를 하는 일련의 구조화된 소프트웨어 공학 활동이다.")
    func.image_resize('requirements_engineering.png', __file__, 500)

with tab[1]:
    st.subheader('Functional Requirements')

    # ['RSQ-ID', '구분', '요구사항명', '요구사항 내용', '날짜', '작성자', '필수 여부']
    df_fr = [
        ['SFR-001', '기능', 'N/A', 'N/A', '24/00/00', '윤선준', '필수'],
        ['SFR-002', '기능', 'N/A', 'N/A', '24/00/00', '윤선준', '필수'],
        ['SFR-003', '기능', 'N/A', 'N/A', '24/00/00', '윤선준', '필수'],
        ['SFR-004', '기능', 'N/A', 'N/A', '24/00/00', '윤선준', '필수'],
        ['SFR-005', '기능', 'N/A', 'N/A', '24/00/00', '윤선준', '필수']
    ]

    st.text("인공지능 모델 기능 요구사항")

    st.table(df_fr)

with tab[2]:
    st.subheader('Non-Functional Requirements')
    st.text("인공지능 모델 비기능 요구사항")