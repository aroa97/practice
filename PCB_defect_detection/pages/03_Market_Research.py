import streamlit as st
from streamlit_function import func

import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar
import streamlit.components.v1 as components

func.set_title(__file__)

tab = st.tabs(['시장조사','OKR'])

with tab[0]:
    radio_market = st.radio(label="market", label_visibility='collapsed', horizontal=True, 
                            options=['국내 PCB 시장전망', '세계 PCB 시장규모', '세계 PCB 분야별 시장전망', '세계 TOP11 PCB 제조업체 매출규모','동사의 주요 고객'])

    if radio_market == '국내 PCB 시장전망':
        st.markdown("연간 평균적으로 3.24%의 상승률을 보이고 있다.")
        c = (Bar()
            .add_xaxis(["2019년", "2020년", "2021년", "2022년", '2023년', '2024년'])
            .add_yaxis('(단위: 억 원)', [73168, 74733, 75429, 78180, 81794, 85815])
            .set_global_opts(title_opts=opts.TitleOpts(title=f"PCB Market", subtitle=""),
                            toolbox_opts=opts.ToolboxOpts())
            .render_embed() # generate a local HTML file
        )
        components.html(c, width=1000, height=500)

        col1, _ = st.columns(2)
        with col1:
            with st.expander("Source"):
                st.image("./streamlit_images/market_research/pcb_market_01.png", use_column_width=True)

    elif radio_market == '세계 PCB 시장규모':

        c = (Bar()
            .add_xaxis(['중국', '일본', '한국', '대만', '북미', '유럽', '기타 아시아'])
            .add_yaxis('2011년', [23844, 9907, 6825, 7820, 3622, 2516, 3464])
            .add_yaxis('2012년', [25530, 9638, 7792, 8195, 3685, 2567, 3660])
            .add_yaxis('2013년', [26551, 9485, 8670, 8555, 3708, 2530, 4000])
            .add_yaxis('2014년', [27878, 9350, 9270, 8777, 3700, 2600, 4300])

            .set_global_opts(title_opts=opts.TitleOpts(title=f"PCB Market", subtitle=""),
                             toolbox_opts=opts.ToolboxOpts())
            .render_embed() # generate a local HTML file
        )
        components.html(c, width=1000, height=500)
        
        col1, _ = st.columns(2)
        with col1:
            with st.expander("Source"):
                st.image("./streamlit_images/market_research/pcb_market_03.png", use_column_width=True)

    elif radio_market == '세계 PCB 분야별 시장전망':
        
        col1, _, col2 = st.columns(3)

        with col1:
            st.text("최소 2.37%, 최대 6.49%의 상승률을 보이고 있다.")
            c = (Bar()
                .add_xaxis([0,1,2,3,4,5,6,7])
                .add_yaxis('2018년', [174, 149, 76, 84, 75, 26, 29, 12])
                .add_yaxis('2019년', [172, 145, 81, 80, 68, 27, 26, 13])
                .add_yaxis('2020년', [165, 150, 97, 75, 62, 27, 26, 13])
                .add_yaxis('2024년', [225, 165, 111, 96, 86, 30, 31, 15])

                .set_global_opts(title_opts=opts.TitleOpts(title=f"PCB Market", subtitle=""),
                                toolbox_opts=opts.ToolboxOpts())
                .render_embed() # generate a local HTML file
            )
            components.html(c, width=1000, height=500)
        with col2:
            st.dataframe(["커뮤니케이션", "컴퓨팅", "반도체 패키지", "소비가전", '전장', '국방·항공', '산업', '의료'], hide_index=False)

        col3, _ = st.columns(2)
        with col3:
            with st.expander("Source"):
                st.image("./streamlit_images/market_research/pcb_market_02.png", use_column_width=True)

    elif radio_market == '세계 TOP11 PCB 제조업체 매출규모':
        col1, _, col2 = st.columns(3)

        with col1:
            c = (Bar()
                .add_xaxis([0,1,2,3,4,5,6,7,8,9,10])
                .add_yaxis('2011년', [2106,1150,2179,1282,2110,1148,978,1370,1179,800,700])
                .add_yaxis('2012년', [2314,1512,2453,1446,2149,1391,1226,1396,1429,955,871])
                .add_yaxis('2013년', [2606,1873,2278,1866,1738,1889,1737,1321,1349,1079,1185])
                .add_yaxis('2014년', [2529,2158,2013,1760,1567,1721,2082,1370,1368,1180,1315])

                .set_global_opts(title_opts=opts.TitleOpts(title=f"PCB Market", subtitle=""),
                                toolbox_opts=opts.ToolboxOpts())
                .render_embed() # generate a local HTML file
            )
            components.html(c, width=1000, height=500)

        with col2:
            st.dataframe(['Nippon Mektron(일본)', 'Zhen Ding(대만)', 'Unimicron(대만)', '삼성전기(한국)', 'Ibiden(일본)', 'Hannstar(대만)', 
                          '영풍그룹(한국)', 'Tripod(대만)', 'TTM(미국)', '대덕그룹(한국)', 'Sumitomo'], hide_index=False)
        
        col3, _ = st.columns(2)
        with col3:
            with st.expander("Source"):
                st.image("./streamlit_images/market_research/pcb_market_04.png", use_column_width=True)

    elif radio_market == '동사의 주요 고객':
        func.image_resize('main_customers.png', __file__, 400)

with tab[1]:
    st.subheader('Objectives and Key Results')

    col1, col2, col3, _ = st.columns(4)

    with col1:
        st.markdown("### :red[Objective 목표]")
        st.text("PCB 불량 검출에 탁월한 모델을 만든다")
    with col2:
        st.markdown("### :red[Key Results 핵심 결과]")
        st.text("1. 6개 이상의 결함 감지")
        st.text("2. 불량 검출 정확도 98% 이상")
    with col3:
        st.markdown("### :red[Initiatives 해야 할 일]")
        st.text("1. PCB 불량 검출 정확도 평가지표 선정")
        st.text("2. PCB 훈련·테스트 데이터 수집")
        st.text("3. WER, CER 측정 후 파라미터 수정")

        
