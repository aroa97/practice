import streamlit as st
from streamlit_function import func

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="마케팅 전략", layout="wide")

st.title("마케팅 전략")

tab = st.tabs(['도출 결과', '데이터 분석', "예측 모델", '마케팅'])

with tab[0]:
    st.text('멤버십 기간, 전 달 이용횟수, 정기이용여부')
with tab[1]:
    st.subheader("EDA")

    radio_eda = st.radio(label="eda", label_visibility='collapsed', horizontal=True, 
                         options=["과정", "데이터 로드", "기초 통계량 확인", "결측치와 이상치 탐색", "변수 간의 관계 시각화", "변수 변환 및 생성"])
    
    df = pd.read_csv("./data/gym_churn_us.csv")

    if radio_eda == "과정":
        st.markdown("""
**EDA 과정**
\n **1. 데이터 로드**
\n- 데이터셋을 불러옵니다. (pandas를 사용하여 CSV, Excel 파일 등을 로드)

\n **2. 기초 통계량 확인**
\n- describe() 메서드를 사용하여 기초 통계량(평균, 표준편차, 최소값, 최대값 등)을 확인합니다.
\n- 데이터의 요약 정보를 확인하여 변수의 분포를 파악합니다.
                
\n **3. 결측치와 이상치 탐색**
\n- isnull() 또는 info() 메서드를 사용하여 결측치를 확인합니다.
\n- 시각화 기법(박스플롯, 히스토그램 등)을 사용하여 이상치를 찾아냅니다.
                    
\n **4. 변수 간의 관계 시각화**
\n- 산점도, 히스토그램, 바이올린 플롯, 상관 행렬 등을 사용하여 변수 간의 관계를 탐색합니다.
\n- seaborn과 matplotlib을 활용하여 시각적으로 표현합니다.
                                    
\n **5. 변수 변환 및 생성**
\n- 필요에 따라 변수를 변환하거나 새로운 변수를 생성하여 데이터 분석의 깊이를 더합니다.""")
    elif radio_eda == "데이터 로드":
        st.link_button(label='Source', url='https://www.kaggle.com/datasets/adrianvinueza/gym-customers-features-and-churn')
        st.code("""
# python jupyter notebook
import pandas as pd
                
df = pd.read_csv('./data/gym_churn_us.csv')
df.head()
""")

        # col1, col2 = st.columns(2)

        # with col1:
        #     df = pd.read_csv("./data/gym_churn_us.csv")
        #     st.dataframe(df.head())
        # with col2:
            # col_df = [["성별 (Gender)", "0 : 여자, 1 : 남자"],
            #           ["근처 위치 (Near_Location)", "사용자가 체육관이 위치한 동네에서 살거나 일하는지 여부"],
            #           ["파트너 (Partner)", "사용자가 관련된 회사에서 일하는지 여부 (체육관은 제휴된 회사의 직원이 할인을 받을 수 있도록 정보를 저장합니다)"],
            #           ["프로모션 친구 (Promo_friends)", "사용자가 친구의 프로모션 코드를 사용하여 처음 가입했는지 여부"],
            #           ["전화번호 (Phone)", "사용자가 전화번호를 제공했는지 여부"],
            #           ["계약 기간 (Contract_period)", "1, 6, 12개월 순으로 나누어져 있음"],
            #           ["그룹 방문 여부 (Group_visits)", "그룹, 개인 여부"],
            #           ["총 평균 추가 요금 (Avg_additional_charges_total)", ""],
            #           ["계약 종료까지 남은 월 수 (Month_to_end_contract)", ""],
            #           ["나이 (Age)", ""],
            #           ["가입 기간 (Lifetime)", "사용자가 체육관에 처음 등록한 이후 경과한 시간(개월 단위)"],
            #           ["총 평균 수업 빈도 (Avg_class_frequency_total)",""],
            #           ["현재 월 평균 수업 빈도 (Avg_class_frequency_current_month)",""],
            #           ["고객 이탈 (Churn)", "회원 탈퇴 여부"]]
            # st.dataframe(col_df)

        col_df = [["성별",
                   "근처 위치",
                   "파트너",
                   "프로모션 친구",
                   "전화번호",
                   "계약 기간",
                   "그룹 방문 여부",
                   "총 평균 추가 요금 ",
                   "계약 종료까지 남은 월 수",
                   "나이",
                   "가입 기간",
                   "총 평균 수업 빈도 (Avg_class_frequency_total)",
                   "현재 월 평균 수업 빈도 (Avg_class_frequency_current_month)",
                   "고객 이탈"],
                   ["0 = 여자, 1 = 남자",
                    "회원 체육관이 위치한 동네에서 살거나 일하는지 여부",
                    "회원 관련된 회사에서 일하는지 여부 (체육관은 제휴된 회사의 직원이 할인을 받을 수 있도록 정보를 저장합니다)",
                    "회원 친구의 프로모션 코드를 사용하여 처음 가입했는지 여부",
                    "회원 전화번호를 제공했는지 여부",
                    "1, 6, 12개월 순으로 나누어져 있음",
                    "그룹, 개인 여부",
                    "",
                    "",
                    "",
                    "회원 체육관에 처음 등록한 이후 경과한 시간(개월 단위)",
                    "",
                    "",
                    "회원 탈퇴 여부"]]
        col_df = pd.DataFrame(col_df, columns=["Gender", "Near_Location", "Partner", "Promo_friends", "Phone", "Contract_period", "Group_visits", "Avg_additional_charges_total", "Month_to_end_contract", "Age", "Lifetime", "Avg_class_frequency_total", "Avg_class_frequency_current_month", "Churn"])
        st.dataframe(col_df, hide_index=True)

        
        st.dataframe(df.head())

    elif radio_eda == "기초 통계량 확인":
        st.code("""
# python jupyter notebook
df.describe()
""")
        st.dataframe(df.describe())

    elif radio_eda == "결측치와 이상치 탐색":
        with st.expander("결측치"):
            st.code("""
# python jupyter notebook
# 결측치 확인
df.info()
df.isnull().sum()
# 중복 확인
df.duplicated().sum()
            """)
            col1, col2 = st.columns(2)
            with col1:
                st.image("./streamlit_images/marketing_strategy/data_info.png", use_column_width=True)
            with col2:
                func.image_resize("data_isnull_sum.png", __file__, 500)
           
        with st.expander("이상치"):
            st.code("""
# python jupyter notebook
# 박스 플롯으로 이상치 확인
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.show()
            """)
            plt.clf()
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df)
            plt.xticks(rotation=45)
            st.pyplot(plt)
        with st.expander("이상치 제거"):
            st.code("""
# python jupyter notebook
                    
# 0.25 0.75 분위수 계산
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# 이상치 범위 정의
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치를 제외한 값만 dataframe에 넣기
df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# 박스플롯 시각화로 잘 제거되었는지 확인
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.show()
            """)
            # 0.25 0.75 분위수 계산
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1

            # 이상치 범위 정의
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 이상치를 제외한 값만 dataframe에 넣기
            # pandas의 DataFrame.any() 메서드는 데이터프레임 내의 값이 True인지 여부를 확인하는 데 사용됩니다.
            df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

            plt.clf()

            # 박스플롯 시각화로 잘 제거되었는지 확인
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df)
            plt.xticks(rotation=45)
            st.pyplot(plt)

    elif radio_eda == "변수 간의 관계 시각화":
        radio_plot = st.radio(label="plot", label_visibility='collapsed', horizontal=True, 
                              options=["Pair Plot", "Bar", "Pie", "Histogram", "Violin Plot", "Linear, Scatter", "Etc", "Corrcoef"])
        if radio_plot == "Pair Plot":
            with st.expander("Source"):
                st.code(""" 
# python jupyter notebook
sns.pairplot(df)
plt.show()
                """)
            st.image("./streamlit_images/marketing_strategy/seaborn_pairplot.png")

        elif radio_plot == "Bar":
            with st.expander("Source"):
                st.code("""
g = sns.FacetGrid(df, col="Churn", height=4, aspect=1.5)
g.map(plt.hist, "Age")
plt.show()

# 총 평균 추가 요금 (Avg_additional_charges_total)
g.map(plt.hist, "Avg_additional_charges_total")
plt.show()

# 가입 기간 (Lifetime) - 사용자가 체육관에 처음 등록한 이후 경과한 시간(개월 단위).
g.map(plt.hist, "Lifetime")
plt.show()
                """)
            plt.clf()

            g = sns.FacetGrid(df, col="Churn", height=4, aspect=1.5)
            g.map(plt.hist, "Age")
            st.pyplot(plt)

            # 총 평균 추가 요금 (Avg_additional_charges_total)
            g.map(plt.hist, "Avg_additional_charges_total")
            st.pyplot(plt)

            # 가입 기간 (Lifetime) - 사용자가 체육관에 처음 등록한 이후 경과한 시간(개월 단위).
            g.map(plt.hist, "Lifetime")
            st.pyplot(plt)
        
        elif radio_plot == "Pie":
            with st.expander("Source"):
                st.code("""
# 이탈 분포 계산
churn_counts = df['Churn'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 막대 차트
axes[0].bar(churn_counts.index, churn_counts.values)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Retained', 'Churned'])
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Number of Customers')
axes[0].set_title('Bar Chart')

# 원형 차트
axes[1].pie(churn_counts.values, labels=['Retained', 'Churned'], autopct='%1.1f%%', startangle=90)
axes[1].set_title('Pie Chart')

# 레이아웃 조정
fig.suptitle('Customer Churn Distribution Overview', fontsize=16)
plt.tight_layout()
plt.show()
                """)
            plt.clf()

            # Calculate churn distribution
            churn_counts = df['Churn'].value_counts()

            # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Bar Chart
            axes[0].bar(churn_counts.index, churn_counts.values)
            axes[0].set_xticks([0, 1])
            axes[0].set_xticklabels(['Retained', 'Churned'])
            axes[0].set_xlabel('Churn')
            axes[0].set_ylabel('Number of Customers')
            axes[0].set_title('Bar Chart')

            # Pie Chart
            axes[1].pie(churn_counts.values, labels=['Retained', 'Churned'], autopct='%1.1f%%', startangle=90)
            axes[1].set_title('Pie Chart')

            # Adjust layout
            fig.suptitle('Customer Churn Distribution Overview', fontsize=16)
            plt.tight_layout()
            st.pyplot(plt)

        elif radio_plot == "Histogram":
            with st.expander("Source"):
                st.code("""
# 나이 분포
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 히스토그램
axes[0].hist(df['Age'], bins=20)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Age Distribution Histogram')

# 박스 플롯
axes[1].boxplot(df['Age'])
axes[1].set_xlabel('Age')
axes[1].set_title('Age Distribution Box Plot')

# 레이아웃 조정
fig.suptitle('Customer Age Distribution', fontsize=16)
plt.tight_layout()
plt.show()
                """)
            plt.clf()

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].hist(df['Age'], bins=20)
            axes[0].set_xlabel('Age')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Age Distribution Histogram')

            axes[1].boxplot(df['Age'])
            axes[1].set_xlabel('Age')
            axes[1].set_title('Age Distribution Box Plot')

            fig.suptitle('Customer Age Distribution', fontsize=16)
            plt.tight_layout()
            st.pyplot(plt)
        
        elif radio_plot == "Violin Plot":
            with st.expander("Source"):
                st.code("""
# 계약 기간 분석

# 막대 차트: 계약 기간 빈도
contract_counts = df['Contract_period'].value_counts()

# Set color palette
sns.set_palette("Set2")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 막대 차트
axes[0].bar(contract_counts.index, contract_counts.values, color=sns.color_palette("Set2"))
axes[0].set_xlabel('Contract Period (Months)')
axes[0].set_ylabel('Number of Customers')
axes[0].set_title('Contract Period Frequency')
axes[0].grid(True, linestyle='--', alpha=0.7)

# 바이올린 플롯 with hue
# hue로 지정한 변수를 기준으로 색상을 변경하여 서로 다른 그룹을 시각화
sns.violinplot(x='Contract_period', y='Age', data=df, ax=axes[1], hue='Contract_period', palette="coolwarm", dodge=False, legend=False)
axes[1].set_xlabel('Contract Period (Months)')
axes[1].set_ylabel('Age')
axes[1].set_title('Age Distribution by Contract Period')
axes[1].grid(True, linestyle='--', alpha=0.7)

# 레이아웃 조정 및 타이틀 추가
fig.suptitle('Contract Period Analysis', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
                """)

            plt.clf()
            # Contract Period Analysis

            # Bar Chart: Contract Period Frequency
            contract_counts = df['Contract_period'].value_counts()

            # Set color palette
            sns.set_palette("Set2")

            # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Bar Chart
            axes[0].bar(contract_counts.index, contract_counts.values, color=sns.color_palette("Set2"))
            axes[0].set_xlabel('Contract Period (Months)')
            axes[0].set_ylabel('Number of Customers')
            axes[0].set_title('Contract Period Frequency')
            axes[0].grid(True, linestyle='--', alpha=0.7)

            # Violin Plot with hue
            sns.violinplot(x='Contract_period', y='Age', data=df, ax=axes[1], hue='Contract_period', palette="coolwarm", dodge=False, legend=False)
            axes[1].set_xlabel('Contract Period (Months)')
            axes[1].set_ylabel('Age')
            axes[1].set_title('Age Distribution by Contract Period')
            axes[1].grid(True, linestyle='--', alpha=0.7)

            # Adjust layout and add a main title
            fig.suptitle('Contract Period Analysis', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            st.pyplot(plt)

        elif radio_plot == "Linear, Scatter":
            with st.expander("Source"):
                st.code("""
# 총 평균 수업 빈도(Avg_class_frequency_total) 분석

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 선형 플롯: 가입 기간(Lifetime)의 총 평균 수업 빈도
sns.lineplot(x='Lifetime', y='Avg_class_frequency_total', data=df, ax=axes[0])
axes[0].set_xlabel('Lifetime (Months)')
axes[0].set_ylabel('Average Class Frequency')
axes[0].set_title('Average Class Frequency Over Lifetime')

# 산점도 플롯: 수업 빈도(Class Frequency) vs. 고객 이탈(Churn)
sns.scatterplot(x='Avg_class_frequency_total', y='Churn', data=df, ax=axes[1])
axes[1].set_xlabel('Average Class Frequency')
axes[1].set_ylabel('Churn')
axes[1].set_title('Class Frequency vs. Churn')

# 레이아웃 조정
fig.suptitle('Average Class Frequency Analysis', fontsize=16)
plt.tight_layout()
plt.show()
                """)
            plt.clf()

            # Average Class Frequency Analysis

            # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Line Plot: Average Class Frequency over Lifetime
            sns.lineplot(x='Lifetime', y='Avg_class_frequency_total', data=df, ax=axes[0])
            axes[0].set_xlabel('Lifetime (Months)')
            axes[0].set_ylabel('Average Class Frequency')
            axes[0].set_title('Average Class Frequency Over Lifetime')

            # Scatter Plot: Class Frequency vs. Churn
            sns.scatterplot(x='Avg_class_frequency_total', y='Churn', data=df, ax=axes[1])
            axes[1].set_xlabel('Average Class Frequency')
            axes[1].set_ylabel('Churn')
            axes[1].set_title('Class Frequency vs. Churn')

            # Adjust layout
            fig.suptitle('Average Class Frequency Analysis', fontsize=16)
            plt.tight_layout()
            st.pyplot(plt)
    
        elif radio_plot == "Corrcoef":
            with st.expander("Source"):
                st.code("""
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap=plt.cm.Blues, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
                """)
            plt.clf()

            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap=plt.cm.Blues, fmt=".2f")
            plt.title('Correlation Heatmap')
            st.pyplot(plt)

    elif radio_eda == "변수 변환 및 생성":
        st.code("""
# 변수 제거
df = df.drop(["Phone", "Near_Location"], axis=1, inplace=True)
""")

with tab[2]:
    st.subheader("Predictive Model")

    st.markdown("""
회귀 모델의 정확도를 높이는 과정 
\n 단순선형 -> 다중 or 다항, 규제를 적용, DecisionTreeRegressor -> 앙상블(RF:랜덤포레스트)
""")

with tab[-1]:
    st.subheader("Marketing")

    radio_marketing = st.radio(label='marketing', label_visibility='collapsed', horizontal=True, options=["온라인 마케팅", "오프라인 마케팅"])
    
    if radio_marketing == '온라인 마케팅':
        st.text("홈페이지 서비스, 이벤트, 이메일")

    elif radio_marketing == '오프라인 마케팅':
        st.text("전단지, 프로모션 및 할인")
