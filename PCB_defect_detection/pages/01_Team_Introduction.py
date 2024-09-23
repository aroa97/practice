import streamlit as st
from streamlit_function import func

func.set_title(__file__)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("윤선준")

    st.markdown("**담당업무**")

    with st.expander("More"):
        st.subheader("인적사항")
        st.text("생년월일 : 1997.04.17")
        st.text("이메일 주소 : saro97@naver.com, ehild97@gmail.com")
        st.markdown("**현재 세종교육에서 직업훈련(인공지능) 중**")

        st.markdown("\n")

        st.markdown("주로 사용하는 프로그래밍 언어 : C++, C#, **Python**, Java, Kotlin")
        st.text("사용 가능한 프로그래밍 언어 : JavaScript, dart, Go, Ruby")
        st.text("etc : html, css, Unity")

        st.markdown("\n")

        st.subheader("그 외")

        st.link_button("Go to other homepage", "https://aroa97.github.io/Web/MyWeb/html/Main.html")
        st.image("./streamlit_images/team_introduction/background_image.png", use_column_width=True)

with col2:
    st.subheader("김성일")

    st.markdown("**담당업무**")

    with st.expander("More"):
        st.subheader("인적사항")

with col3:
    st.subheader("왕재권")

    st.markdown("**담당업무**")

    with st.expander("More"):
        st.subheader("인적사항")
