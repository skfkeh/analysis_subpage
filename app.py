# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from PIL import Image     # 이미지 처리 라이브러리

import matplotlib.pyplot as plt 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    return df

### 9.범주형 변수 처리
def preprocess_Dummy(df):
    df = pd.get_dummies(df, columns=['weekday_name','Add_col','Air_col'], drop_first=True)
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


options = st.sidebar.radio('Why is my airfare expensive?!', options=['01. Home','02. 데이터 전처리 과정','03. 알고리즘 적용', '04. 우수 모델 선정'])

# if uploaded_file:
#    df = pd.read_excel(url)

if options == '01. Home':
    st.title('내 항공료는 왜 비싼 것인가')
    st.write('다음 항목은 사이드 메뉴를 확인해 주세요.')

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
    
    
    #### 5. Additional_Info 전처리
    st.header("5. Additional_Info 전처리")
    code_addition = '''# 20 보다 적은 수의 Additional_Info Others 로 변환
add_count = df.Additional_Info.value_counts().index
additional_thing = [l for l in add_count if list(df.Additional_Info).count(l) < 20]
df.Additional_Info = df.Additional_Info.replace(additional_thing, 'Others')

# 건 수가 많은 순서로 변수로 지정
add_count = df.Additional_Info.value_counts().index

# Add_col 컬럼에 인덱스 번호로 넣기
for t in range(len(add_count)):
    df.loc[df.Additional_Info == add_count[t], 'Add_col'] = t'''
    st.code(code_addition, language='python')    
    df = preprocess_Additional(df)
    st.write('')

    
    #### 6. Total_Stops 전처리
    st.header("6. Total_Stops 전처리")
    
    code_Stop = '''def handle_stops(x):
    if x == 'non-stop': return 0
    return int(x.split()[0])`

df.Total_Stops = df.Total_Stops.apply(handle_stops)'''
    st.code(code_Stop, language='python')
    df = preprocess_Stops(df)
    st.dataframe(df.head())
    st.write('')
    
    
    #### 7. Date_of_Journey 전처리
    st.header("7. Date_of_Journey 전처리")
    
    code_Date = '''df['Date_of_journey_DT'] = pd.to_datetime(df['Date_of_Journey'])
df['weekday'] = pd.to_datetime(df['Date_of_journey_DT']).dt.weekday
df['weekday_name'] = pd.to_datetime(df['Date_of_journey_DT']).dt.day_name()'''

    st.code(code_Date, language='python')
    df = preprocess_Date(df)
    st.write('')
    
    
    ### 8. Dep_Time 데이터 전처리
    st.header("8. Dep_Time 전처리")
    code_Dep = '''df.Dep_Time = df.Dep_Time.astype(str)
df['Dep_hour'] = df.Dep_Time.str.extract('([0-9]+)\:')
df.drop(columns=['Dep_Time'],inplace=True)'''
    df = preprocess_Dep_Time(df)    
    st.code(code_Dep, language='python')
    st.dataframe(df.head())
    st.write('')
    
    
    ### 9. 불필요 컬럼 drop
    st.header("9. 불필요 컬럼 drop")
    st.write('분석에 불필요한 컬럼을 Drop 처리한다.')
    st.subheader('대상')
    col1, col2, col3 = st.columns(3)
    
    col1.write('- Date_of_Journey')
    col2.write('- Source')
    col3.write('- Additional_Info')
    col1.write('- Date_of_journey_DT')
    col2.write('- Destination')
    col3.write('- weekday')
    df = preprocess_Drop(df)
    st.dataframe(df.head())
    st.write('')
    
    
    ### 10.범주형 변수 처리
    st.header("10.범주형 변수 처리")
    st.write('정리된 column 중 object로 남아있는 column들을 dummy 처리한다.')
    code_Dummy = "df = pd.get_dummies(df, columns=['weekday_name','Add_col','Air_col'], drop_first=True)"
    st.code(code_Dummy, language='python')
    st.write('')
    st.markdown('---')
    st.write('')
    
    st.header('전처리 완료')
    df = preprocess_Dummy(df)
    st.dataframe(df.head())
    
    st.image('https://github.com/skfkeh/newthing/blob/main/img/plane_landing.png?raw=true')

    
################################
#####   Algorithm Start    #####
################################
elif options == '03. 알고리즘 적용':
    st.title("분석 알고리즘에 따른 predict 값 ")

    tab_DT, tab_RF, tab_XGB = st.tabs(["DecisionTree", "RandomForest", "XGBoost"])

    data_path = f"{os.path.dirname(os.path.abspath(__file__))}/data.csv"
    data = pd.read_csv(data_path)
    df = pd.DataFrame(data)
    df.drop('Unnamed: 0',axis=1,inplace=True)
    # 데이터 전처리

    X = df.drop('Price', axis=1)
    y = df.Price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    
    #### Tab1
    with tab_DT:
        st.header("DecisionTree")
        
        # score 와 mse 비교
        model_pkl_path_dt = f"{os.path.dirname(os.path.abspath(__file__))}/decisionTree.pkl"
        model_dt = joblib.load(model_pkl_path_dt)
        
        st.subheader('예측하기')
        
        train_pred_dt = model_dt.predict(X_train) 
        test_pred_dt = model_dt.predict(X_test)
        
#         predict_button_dt = st.button('예측')
        
#         if predict_button_dt:        
        st.write(f'Train-set : {model_dt.score(X_train, y_train)}')
        st.write(f'Test-set : {model_dt.score(X_test, y_test)}')

        # 훈련 모델 시각화
        st.subheader('모델 훈련이 잘 되었는지 시각화')
        r1_col, r2_col = st.columns(2)
        r1_col.image('https://github.com/skfkeh/newthing/blob/main/img/first_pred_dt.png?raw=true',caption='초기 파라미터 값 그래프')
        r2_col.image('https://github.com/skfkeh/newthing/blob/main/img/optim_pred_dt.png?raw=true',caption='최적의 파라미터 적용 그래프')

        # 기본값일 때 결정계수
        st.subheader('RMSE 비교')
        train_relation_square_dt = model_dt.score(X_train, y_train)
        test_relation_square_dt = model_dt.score(X_test, y_test)
        st.write(f'Train 결정계수 : {train_relation_square_dt}')
        st.write(f'Test 결정계수 : {test_relation_square_dt}')
        
        st.subheader('시각화 부분')
        CheckBox_dt = st.checkbox('plotly 활성화')

        if CheckBox_dt:
            fig_dt = make_subplots(rows=1, cols=1, shared_xaxes=True)
            fig_dt.add_trace(go.Scatter(x=y_train,y=y_test, mode='markers_dt',name='Actual'))
            fig_dt.add_trace(go.Scatter(x=y_test, y=test_pred_dt, mode='markers_dt',name='Predict')) # mode='lines+markers'
            fig_dt.update_layout(title='<b>actual과 predict 비교')
            st.plotly_chart(fig_dt)
        
    #### Tab2
    with tab_RF:
        st.header("RandomForest")
        
        # score 와 mse 비교
        model_pkl_path_rf = f"{os.path.dirname(os.path.abspath(__file__))}/randomforest.pkl"
        model_rf = joblib.load(model_pkl_path_rf)

        # 파라미터 변경해가며 예측
        st.subheader('예측하기')
#         s1_col, s2_col, s3_col = st.columns(3)
#         s1_col.selectbox('choose n_estimators',[200,500,800,1000])
#         s2_col.selectbox('choose max_dpth',[5,9,12,20])
#         s3_col.selectbox('choose min_samples_leaf',[5,11,18,22])
        
        train_pred_rf = model_rf.predict(X_train) 
        test_pred_rf = model_rf.predict(X_test)

#         predict_button_rf = st.button('예측')
#         if predict_button_rf:
        st.write(f'Train-set : {model_rf.score(X_train, y_train)}')
        st.write(f'Test-set : {model_rf.score(X_test, y_test)}')
            
        # 훈련 모델 시각화
        st.subheader('모델 훈련이 잘 되었는지 시각화')
        r1_col, r2_col = st.columns(2)
        r1_col.image('data:image/png;base64,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',caption='초기 파라미터 값 그래프')
        r2_col.image('data:image/png;base64,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',caption='최적의 파라미터 적용 그래프')

        st.subheader('RMSE 비교') 
        train_relation_square_rf = mean_squared_error(y_train, train_pred_rf, squared=False)
        test_relation_square_rf = mean_squared_error(y_test, test_pred_rf) ** 0.5
        st.write(f'train 결정계수 : {train_relation_square_rf}')
        st.write(f'test 결정계수 : {test_relation_square_rf}')

        st.subheader('시각화 부분')
        CheckBox_rf = st.checkbox('plotly 활성화')

        if CheckBox_rf:
            # 시각화 해보기
            fig_rf = make_subplots(rows=1, cols=1, shared_xaxes=True)
            fig_rf.add_trace(go.Scatter(x=y_train,y=y_test, mode='markers_rf',name='Actual'))
            fig_rf.add_trace(go.Scatter(x=y_test, y=test_pred_rf, mode='markers_rf',
                        name='Predict')) # mode='lines+markers'
            fig_rf.update_layout(title='<b>actual과 predict 비교')
            st.plotly_chart(fig_rf)

    #### Tab3
    with tab_XGB:
       st.header("XGBoost")
       st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
        
        
elif options == '04. 우수 모델 선정':
    st.title('우수 모델')
