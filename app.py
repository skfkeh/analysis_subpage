# app.py
import streamlit as st
import pandas as pd
import numpy as np

import plotly.figure_factory as ff
# import matplotlib.pyplot as plt 

import time
from PIL import Image     # 이미지 처리 라이브러리


########### function ###########
## 카운트 다운
def count_down(ts):
    with st.empty():
        input_time = 1*3
        while input_time>=0:
            minutes, seconds = input_time//60, input_time%60
            st.metric("Countdown", f"{minutes:02d}:{seconds:02d}")
            time.sleep(1)
            input_time -= 1
        st.empty()

## 데이터 전처리

### 1. Route Drop 처리
def preprocess_Route(df):
    df.drop('Route', axis=1, inplace=True)
    return df

### 2. Duration 전처리
def preprocess_Duration(df):
    df['Dep_Time'] = pd.to_datetime(df['Dep_Time'], format= '%H:%M').dt.time
    df['Duration_hour'] = df.Duration.str.extract('(\d+)h')
    df['Duration_min'] = df.Duration.str.extract('(\d+)m').fillna(0)
    df.drop('Duration', axis=1, inplace=True)
    df.drop(index=6474,inplace=True)

    df.Duration_hour = df.Duration_hour.astype('int64')
    df.Duration_min = df.Duration_min.astype('int64')
    df.Duration_hour = df.Duration_hour*60
    df['Duration_total'] = df.Duration_hour+df.Duration_min
    df.drop(columns=['Duration_hour','Duration_min','Arrival_Time'],inplace=True)
    
    return df

### 3. Airline 전처리
def preprocess_Airline(df):
    air_count = df.Airline.value_counts().index
    airlist = [l for l in air_count if list(df.Airline).count(l) < 200]
    df.Airline = df.Airline.replace(airlist, 'Others')

    for t in range(len(air_count)):
        df.loc[df.Airline == air_count[t], 'Air_col'] = t
    df.drop(columns=['Airline'],inplace=True)
    
    return df

### 4. Additional_Info 전처리
def preprocess_Additional(df):
    add_count = df.Additional_Info.value_counts().index
    additional_thing = [l for l in add_count if list(df.Additional_Info).count(l) < 20]
    df.Additional_Info = df.Additional_Info.replace(additional_thing, 'Others')

    add_count = df.Additional_Info.value_counts().index
    for t in range(len(add_count)):
        df.loc[df.Additional_Info == add_count[t], 'Add_col'] = t
        
    return df

### 5. Total_Stops 전처리
def preprocess_Stops(df):
    df.loc[df.Total_Stops.isna(),'Total_Stops'] = '1 stop'

    def handle_stops(x):
        if x == 'non-stop': return 0
        return int(x.split()[0])

    df.Total_Stops = df.Total_Stops.apply(handle_stops)

    return df
    
### 6. Date_of_Journey 전처리
def preprocess_Date(df):
    df['Date_of_journey_DT'] = pd.to_datetime(df['Date_of_Journey'])
    df['weekday'] = pd.to_datetime(df['Date_of_journey_DT']).dt.weekday
    df['weekday_name'] = pd.to_datetime(df['Date_of_journey_DT']).dt.day_name()
    
    return df

### 7. Dep_Time 데이터 전처리
def preprocess_Dep_Time(df):
    df.Dep_Time = df.Dep_Time.astype(str)
    df['Dep_hour'] = df.Dep_Time.str.extract('([0-9]+)\:')
    df.drop(columns=['Dep_Time'],inplace=True)
    
    return df

### 8. 불필요 컬럼 drop
def preprocess_Drop(df):
    df.drop(columns=['Date_of_Journey',
                     'Source','Destination',
                     'Date_of_journey_DT',
                     'Additional_Info',
                     'weekday'],inplace=True)

### 9.범주형 변수 처리
def preprocess_Dummy(df):
    df = pd.get_dummies(df, columns=['weekday_name','Add_col','Air_col'],drop_first=True)
    return df
    
########### function ###########
        
    
########### session ###########

if 'chk_balloon' not in st.session_state:
    st.session_state['chk_balloon'] = False

if 'chk_strline' not in st.session_state:
    st.session_state['chk_strline'] = ''

if 'choice' not in st.session_state:
    st.session_state['choice'] = ''

if 'file_name' not in st.session_state:
    st.session_state['file_name'] = ''
    
########### session ###########
       

########### define ###########

file_name = 'Data_Train.csv'
url = f'https://raw.githubusercontent.com/skfkeh/newthing/main/{file_name}'

########### define ###########




################################
#####       UI Start       #####
################################

#if st.session_state['chk_balloon'] == False:
#    count_down(5)
#    with st.spinner(text="Please wait..."):
#        time.sleep(1)

#    st.balloons()
#    st.session_state['chk_balloon'] = True


options = st.sidebar.radio('Why is my airfare expensive?!', options=['01. Home','02. 데이터 전처리 과정','03. 시각화(plotly)'])

# if uploaded_file:
#    df = pd.read_excel(url)

if options == '01. Home':
    st.title('내 항공료는 왜 비싼 것인가')
    st.header('다음 항목은 사이드 메뉴를 확인해 주세요.')

    jpg_url = "https://github.com/skfkeh/MachineLearning/blob/main/img/why.png?raw=true"
    # st.set_page_config(layout="wide")
    st.image(jpg_url, caption="Why So Serious??!")

    st.write(f"사용한 데이터 URL : {url}")

elif options == '02. 데이터 전처리 과정':
    st.image('https://github.com/skfkeh/MachineLearning/blob/main/img/plane_img.png?raw=true')
    df = pd.read_csv(url)

    ### 1. df.head()로 데이터 확인
    st.header("1. df.head()로 데이터 확인")
    st.dataframe(df.head())
    st.write('')
    

    ### 2. Route Drop 처리
    st.header("2. Route Drop 처리")
    st.write('Total_Stops 와 Route가 관련 있는 컬럼인가')
    st.write(' => stop 수에 맞춰 Route가 늘어나는 것 확인')
    st.write('가장 먼저 Route를 drop 처리 => Total_Stops 라는 컬럼으로 활용')
    code_Route = '''df.drop('Route', axis=1, inplace=True)'''
    st.code(code_Route, language='python')
    st.write('')
    df = preprocess_Route(df)


    ### 3. Duration 전처리
    st.header("3. Duration 전처리")
    code_Dep = '''#Duration 컬럼을 '시간'과 '분' 단위로 분할
df['Dep_Time'] = pd.to_datetime(df['Dep_Time'], format= '%H:%M').dt.time
df['Duration_hour'] = df.Duration.str.extract('(\d+)h')
df['Duration_min'] = df.Duration.str.extract('(\d+)m').fillna(0)

# 계산을 위해 str -> int64 로 변경하고 Duration_hour을 분으로 변경
# 최종적으로 Duration_total 컬럼으로 내린다
df.Duration_hour = df.Duration_hour.astype('int64')
df.Duration_min = df.Duration_min.astype('int64')
df.Duration_hour = df.Duration_hour*60
df['Duration_total'] = df.Duration_hour+df.Duration_min'''
    st.code(code_Dep, language='python')
    
    df = preprocess_Duration(df)

    st.dataframe(df.head())
    st.write('')


    #### 4. Airline 전처리
    st.header("4. Airline 전처리")
    code_airline = '''air_count = df.Airline.value_counts().index
# 200 보다 적은 수의 airline은 Others 로 변환
airlist = [l for l in air_count if list(df.Airline).count(l) < 200]
df.Airline = df.Airline.replace(airlist, 'Others')

# Air_col : Airline을 번호로 분류
for t in range(len(air_count)):
    df.loc[df.Airline == air_count[t], 'Air_col'] = t'''
    st.code(code_airline, language='python')
    df = preprocess_Airline(df)
    
    st.write('')
    
    
    #### 5. Total_Stops 전처리
    st.header("5. Total_Stops 전처리")
    
    code_Stop = '''def handle_stops(x):
    if x == 'non-stop': return 0
    return int(x.split()[0])

df.Total_Stops = df.Total_Stops.apply(handle_stops)'''
    st.code(code_Stop, language='python')
    df = preprocess_Stops(df)
    st.dataframe(df.head())
    st.write('')
    
    
    #### 6. Date_of_Journey 전처리
    st.header("6. Date_of_Journey 전처리")
    
    code_Date = '''df['Date_of_journey_DT'] = pd.to_datetime(df['Date_of_Journey'])
df['weekday'] = pd.to_datetime(df['Date_of_journey_DT']).dt.weekday
df['weekday_name'] = pd.to_datetime(df['Date_of_journey_DT']).dt.day_name()'''

    st.code(code_Date, language='python')
    df = preprocess_Date(df)
    st.write('')
    
    
    ### 7. Dep_Time 데이터 전처리
    st.header("7. Dep_Time 전처리")
    code_Dep = '''df.Dep_Time = df.Dep_Time.astype(str)
df['Dep_hour'] = df.Dep_Time.str.extract('([0-9]+)\:')
df.drop(columns=['Dep_Time'],inplace=True)'''
    df = preprocess_Dep_Time(df)    
    st.code(code_Dep, language='python')
    st.dataframe(df.head()) 
    st.write('')
    
    
    ### 8. 불필요 컬럼 drop
    st.header("8. 불필요 컬럼 drop")
    st.write('분석에 불필요한 컬럼을 Drop 처리한다.')
    st.header('대상')
    st.write('- Date_of_Journey')
    st.write('- Source')
    st.write('- Destination')
    st.write('- Date_of_journey_DT')
    st.write('- Additional_Info')
    st.write('- weekday')
    df = preprocess_Drop(df)
    st.dataframe(df.head()) 
    st.write('')
    
    
    ### 9.범주형 변수 처리
    st.header("9.범주형 변수 처리")
    st.write('정리된 column 중 object로 남아있는 column들을 dummy 처리한다.')
    code_Dummy = "df = pd.get_dummies(df,columns=['weekday_name','Add_col','Air_col'],drop_first=True)"
    st.code(code_Dummy, language='python')
    
    st.dataframe(df)
    
    st.title('전처리 완료')
    
elif options == '03. 시각화(plotly)':
    st.write("분석 알고리즘을 골라주세요")

    tab_De, tab_RF, tab_XGB = st.tabs(["DecisionTree", "RandomForest", "XGBoost"])

    #### Tab1
    with tab_De:
       col1, col2 = st.columns(2)

       st.header("Logistic")
       st.image("https://github.com/skfkeh/newthing/blob/main/img/Patrick.jpeg?raw=true", width=200)

       ts_number = col1.slider(label="test_size를 설정해주세요",
                              min_value=0.00, max_value=1.00,
                              step=0.10, format="%f")

       rs_number = col2.slider(label="random_state 설정",
                                  min_value=0, max_value=200,
                                  step=50, format="%d")

       # st.write(f'Test_size : {ts_number}      Random_state : {rs_text}{rs_number}')

    #### Tab2
    with tab_RF:
       st.header("RandomForest")
       st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    #### Tab3
    with tab_XGB:
       st.header("XGBoost")
       st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

        
        
    SearchBtn = st.button('Search')
    
    if SearchBtn:
        # Add histogram data
    #     x0 = np.random.randn(200) - 5
        x1 = np.random.randn(200) - 2
        x2 = np.random.randn(200)
        x3 = np.random.randn(200) + 2
    #     x4 = np.random.randn(200) + 5

        # Group data together
        hist_data = [x1, x2, x3]

        group_labels = ['Group 1', 'Group 2', 'Group 3']

        # Create distplot with custom bin_size
        fig = ff.create_distplot(
               hist_data, group_labels, bin_size=[.1, .25, .5])

        # Plot!
        st.plotly_chart(fig, use_container_width=True)
