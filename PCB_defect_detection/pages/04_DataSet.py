import streamlit as st
from streamlit_function import func

func.set_title(__file__)

tab = st.tabs(["흑백 데이터셋", "실물 데이터셋"])

with tab[0]:
    st.page_link('./pages/98_Source.py', label='Source', icon="🚨")

    col1, col2, col3 = st.columns(3)
    img_size = 350
    with col1:
        st.text('정상 PCB : 1500개')
        func.image_resize('pcb_01.jpg',__file__, img_size)
        func.image_resize('pcb_02.jpg',__file__, img_size)
        func.image_resize('pcb_03.jpg',__file__, img_size)
        
    with col2:
        st.text('불량 PCB : 1500개')
        func.image_resize('pcb_defect_01.jpg',__file__, img_size)
        func.image_resize('pcb_defect_02.jpg',__file__, img_size)
        func.image_resize('pcb_defect_03.jpg',__file__, img_size)
        
    with col3:
        st.text('절대값 차 연산')
        func.image_resize('diff_01.png',__file__, img_size)
        func.image_resize('diff_02.png',__file__, img_size)
        func.image_resize('diff_03.png',__file__, img_size)

with st.expander("Source Code"):
    st.code("""
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img1 = cv2.imread('img1_path')
img2 = cv2.imread('img2_path')

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(img1_gray, img2_gray)

_, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)

plt.imshow(diff,'gray')
plt.xticks([]),plt.yticks([])
plt.show()
    """)

with tab[1]:
    st.page_link('./pages/98_Source.py', label='Source', icon="🚨")
    col1, col2, col3 = st.columns(3)

    with col1:
        func.image_resize('07.jpg',__file__, 250)
        st.text('정상 PCB : 12개')
    with col2:
        func.image_resize('pcb_defect_marking_01.png',__file__, 250)
        st.text('불량 PCB : 693개')
    with col3:
        func.image_resize('pcb_defect_marking_02.png',__file__, 250)
        st.text('불량 PCB(회전) : 693개')
