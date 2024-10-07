import streamlit as st
from streamlit_function import func
import pandas as pd

func.set_title(__file__)

tab = st.tabs(['요구공학', '기능 요구사항', '비기능 요구사항'])

with tab[0]:
    st.subheader('Requirements Engineering')
    st.text("요구공학은 시스템 개발에 필요한 요구사항을 도출하여 문서화(명세화)하고 이를 체계적 으로 검증 및 관리를 하는 일련의 구조화된 소프트웨어 공학 활동이다.")
    func.image_resize('requirements_engineering.png', __file__, 500)

with tab[1]:
    st.subheader('Functional Requirements')
    st.text("인공지능 모델 기능 요구사항")

    columns = ['RSQ-ID', '구분', '요구사항명', '요구사항 내용', '날짜', '작성자', '필수 여부']
    df_fr = [
        ['SFR-001', '기능', 'Open 결함 검출', '회로 개방 결함을 검출한다', '24/09/00', '윤선준', '필수'],
        ['SFR-002', '기능', 'Short 결함 검출', '회로 단락 결함을 검출한다', '24/09/00', '윤선준', '필수'],
        ['SFR-003', '기능', 'Copper 결함 검출', '구리 결함 결함을 검출한다', '24/09/00', '윤선준', '필수'],
        ['SFR-004', '기능', 'Mousebit 결함 검출', '마우스바이트 결함을 검출한다', '24/09/00', '윤선준', '필수'],
        ['SFR-005', '기능', 'Pin-hole 결함 검출', '핀홀 결함을 검출한다', '24/09/00', '윤선준', '필수'],
        ['SFR-006', '기능', 'Spur 결함 검출', '스퍼 결함을 검출한다', '24/09/00', '윤선준', '필수'],
        ['SFR-007', '기능', '결함 표시', '검출된 결함을 PCB 이미지에 표시한다', '24/09/00', '윤선준', '선택'],
        ['SFR-008', '기능', '결함 이미지 저장', '데이터로 활용하기 위해 데이터베이스 공간에 저장한다', '24/09/00', '윤선준', '필수'],
        ['SFR-009', '기능', '화재 경보', '화재 발생 시 119에 연락 및 경보를 울린다', '24/09/00', '윤선준', '필수']
    ]
    df_fr = pd.DataFrame(df_fr, columns=columns)

    st.dataframe(df_fr, use_container_width=True, hide_index=True)

with tab[2]:
    st.subheader('Non-Functional Requirements')
    st.text("인공지능 모델 비기능 요구사항")

    columns = ['RSQ-ID', '구분', '요구사항명', '요구사항 내용', '날짜', '작성자', '필수 여부']
    df_nfr = [
        ['PFR-001', '비기능', '속도', 'FPS 30 이상', '24/09/00', '윤선준', '필수'],
        ['PFR-002', '비기능', '데이터양', '학습 데이터 1만장 이상', '24/09/00', '윤선준', '필수'],
        ['PFR-003', '비기능', '정확도', 'mAP 90% 이상', '24/09/00', '윤선준', '선택'],
        ['PFR-004', '비기능', 'TOP-5', 'TOP-5 정확도 95%이상', '24/09/00', '윤선준', '필수']
    ]
    df_nfr = pd.DataFrame(df_nfr, columns=columns)

    st.dataframe(df_nfr, use_container_width=True, hide_index=True)