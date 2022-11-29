# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import graphviz
from PIL import Image     # 이미지 처리 라이브러리

import matplotlib.pyplot as plt 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import random       # 각 plotly 그래프의 key값을 적용하기 위한 import

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

    
# if 'rmse_train_dt' not in st.session_state:
#     st.session_state['rmse_train_dt'] = 0
# if 'rmse_test_dt' not in st.session_state:
#     st.session_state['rmse_test_dt'] = 0
# if 'rmse_train_rf' not in st.session_state:
#     st.session_state['rmse_train_rf'] = 0
# if 'rmse_test_rf' not in st.session_state:
#     st.session_state['rmse_test_rf'] = 0
########### session ###########
       

########### define ###########
file_name = 'Data_Train.csv'
url = f'https://raw.githubusercontent.com/skfkeh/MachineLearning/main/Resource/{file_name}'
keys = random.sample(range(1000, 9999), 3)
########### define ###########


################################
#####       UI Start       #####
################################

if st.session_state['chk_balloon'] == False:
#    count_down(5)
#    with st.spinner(text="Please wait..."):
#        time.sleep(1)

    st.balloons()
    st.snow()
    st.session_state['chk_balloon'] = True

    
options = st.sidebar.radio('Why is my airfare expensive?!', options=['01. Home','02. 데이터 전처리 과정','03. 알고리즘 적용', '04. 우수 모델 선정', '05. Error!!'])

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
    
    st.image('https://github.com/skfkeh/MachineLearning/blob/main/img/plane_landing.png?raw=true')

    
################################
#####   Algorithm Start    #####
################################
elif options == '03. 알고리즘 적용':
    st.title("분석 알고리즘에 따른 predict 값 ")

    st.subheader('Heatmap 기반으로 column 값이 가장 관련 있는지 확인')
    visible_cb = st.checkbox('분석 정보')
    st.write('''
    
    
    
    ''')
    if visible_cb:
        st.image('https://github.com/skfkeh/MachineLearning/blob/main/img/flight_heatmap.jpg?raw=true', caption='Price값은 Duration_total column과 가장 관련이 깊었다.')
        st.write('위 정보를 기반으로 각 알고리즘에 맞춰 분석해보자.')
        st.write('사용된 분석 알고리즘')
        st.write('- Decision Tree')
        st.write('- RandomForest')
        st.write('- XGBoost')
    st.markdown('---')
    
    
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
        r1_col.image('https://github.com/skfkeh/MachineLearning/blob/main/img/first_pred_dt.png?raw=true',caption='초기 파라미터 값 그래프')
        r2_col.image('https://github.com/skfkeh/MachineLearning/blob/main/img/optim_pred_dt.png?raw=true',caption='최적의 파라미터 적용 그래프')

        # 기본값일 때 결정계수
        st.subheader('RMSE 비교')
        train_relation_square_dt = model_dt.score(X_train, y_train)
#         st.session_state['rmse_train_dt'] = model_dt.score(X_train, y_train)
        test_relation_square_dt = model_dt.score(X_test, y_test)
#         st.session_state['rmse_test_dt'] = model_dt.score(X_test, y_test)
        
        st.write(f'Train 결정계수 : {train_relation_square_dt}')
        st.write(f'Test 결정계수 : {test_relation_square_dt}')
        
        st.subheader('시각화 부분')
#         CheckBox_dt = st.checkbox('plotly 활성화')

#         if CheckBox_dt:
        # 시각화 해보기
        fig_dt = make_subplots(rows=1, cols=1, shared_xaxes=True)
        fig_dt.add_trace(go.Scatter(x=y_train,y=y_test, mode='markers',name='Actual_dt'))
        fig_dt.add_trace(go.Scatter(x=y_test, y=test_pred_dt, mode='markers',name='Predict_dt'))
        fig_dt.update_layout(title='<b>actual과 predict 비교_dt')
        st.plotly_chart(fig_dt, key = keys[0])

        st.subheader('Decision Tree')
        st.image('https://github.com/skfkeh/MachineLearning/blob/main/img/dt_graph.png?raw=true')
        
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
        r1_col.image('https://github.com/skfkeh/MachineLearning/blob/main/img/first_pred_rf.png?raw=true',caption='초기 파라미터 값 그래프')
        r2_col.image('https://github.com/skfkeh/MachineLearning/blob/main/img/optim_pred_rf.png?raw=true',caption='최적의 파라미터 적용 그래프')

        st.subheader('RMSE 비교') 
        train_relation_square_rf = mean_squared_error(y_train, train_pred_rf, squared=False)
#         st.session_state['rmse_train_rf'] = mean_squared_error(y_train, train_pred_rf, squared=False)
        test_relation_square_rf = mean_squared_error(y_test, test_pred_rf) ** 0.5
#         st.session_state['rmse_test_rf'] = mean_squared_error(y_test, test_pred_rf) ** 0.5
        st.write(f'train 결정계수 : {train_relation_square_rf}')
        st.write(f'test 결정계수 : {test_relation_square_rf}')

        st.subheader('시각화 부분')
#         CheckBox_rf = st.checkbox('plotly 활성화')

#         if CheckBox_rf:
        # 시각화 해보기
        fig_rf = make_subplots(rows=1, cols=1, shared_xaxes=True)
        fig_rf.add_trace(go.Scatter(x=y_train,y=y_test, mode='markers',name='Actual_rf'))
        fig_rf.add_trace(go.Scatter(x=y_test, y=test_pred_rf, mode='markers', name='Predict_rf')) # mode='lines+markers'
        fig_rf.update_layout(title='<b>actual과 predict 비교_rf')
        st.plotly_chart(fig_rf, key = keys[1])
        

    #### Tab3
    with tab_XGB:
        st.header("XGBoost")
        
        st.image('https://github.com/skfkeh/MachineLearning/blob/main/img/emergency.jpeg?raw=true', caption='모듈의 버전차이로 인해 다른 화면에서 추가 설명')
        # score 와 mse 비교
#         model_pkl_path_xg = f"{os.path.dirname(os.path.abspath(__file__))}/XGBoost_02.pkl"
#         model_xg = joblib.load(model_pkl_path_xg)
        
#         # 파라미터 변경해가며 예측
#         st.subheader('예측하기')
#         train_pred_xg = model_xg.predict(X_train)
#         test_pred_xg = model_xg.predict(X_test)
#         st.write(f'Train-set : {model_xg.score(X_train, y_train)}')
#         st.write(f'Test-set : {model_xg.score(X_test, y_test)}')
        
#         # 훈련 모델 시각화
#         st.subheader('모델 훈련이 잘 되었는지 시각화')
#         r1_col, r2_col = st.columns(2)
#         r1_col.image('https://github.com/skfkeh/MachineLearning/blob/main/img/first_pred_xgb.png?raw=true',caption='초기 파라미터 값 그래프')
#         r2_col.image('https://github.com/skfkeh/MachineLearning/blob/main/img/optim_pred_xgb.png?raw=true',caption='최적의 파라미터 적용 그래프')
        
#         # 기본값일 때 결정계수
#         st.subheader('RMSE 비교')
#         train_relation_square_xg = model_xg.score(X_train, y_train)
#         test_relation_square_xg = model_xg.score(X_test, y_test)
#         st.write(f'Train 결정계수 : {train_relation_square_xg}')
#         st.write(f'Test 결정계수 : {test_relation_square_xg}')
        
#         st.subheader('시각화 부분')
#         # CheckBox_dt = st.checkbox('plotly 활성화')
# #         if CheckBox_dt:
#         # 시각화 해보기
#         fig_xg = make_subplots(rows = 1, cols = 1, shared_xaxes = True)
#         fig_xg.add_trace(go.Scatter(x = y_train, y = y_test, mode = 'markers', name = 'Actual_xg'))
#         fig_xg.add_trace(go.Scatter(x = y_test, y = test_pred_xg, mode = 'markers', name = 'Predict_xg'))
#         fig_xg.update_layout(title = '<b>actual과 predict 비교_xg')
#         st.plotly_chart(fig_xg, key = keys[2])
        
elif options == '04. 우수 모델 선정':
    st.title('우수 모델')

    st.subheader('각 알고리즘의 Train/Test RMSE 간격 차이')    
    
    rmse_dt = pd.DataFrame(
        [['DecisionTree', '2707.9318335945104', '2808.279093924912', 2808.279093924912 - 2707.9318335945104],
         ['RandomForest', '2689.0929933761795', '2743.3126679067127', 2743.3126679067127 - 2689.0929933761795],
         ['XGBoost', '1303.7460823618917', '2654.185010859511', 2654.185010859511 - 1303.7460823618917]],
        columns=['', 'Train', 'Test', 'Train과 Test 간의 차이']
    )
    
    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    
    st.table(rmse_dt)
    
    st.success('XGBoost의 간격이 가장 크므로 가장 효율적이라고 할 수 있다.', icon="✅")
#     st.write('XGBoost의 간격이 가장 크므로 가장 효율적이라고 할 수 있다.')
    
elif options == '05. Error!!':
    st.title('이게 참 어려웠지~(아련)')
    
    st.subheader('1. 데이터 전처리 - Duration')
    st.write('Duration_hour가 데이트 타임 형식으로 변환이 안 되는 오류 발생')
    st.image('https://github.com/skfkeh/MachineLearning/blob/main/img/duration_error_.PNG?raw=true', caption="Duration_hour error")
    st.write('해결책 : 시간과 분을 int형으로 바꿔주고 시간을 분으로 환산해서 Duration_total 컬럼 생성')
    code_duration = '''df.Duration_hour = df.Duration_hour.astype('int64')
df.Duration_min = df.Duration_min.astype('int64')'''
    st.code(code_duration, language='python')
    st.markdown('---')
    
    st.subheader('2. 훈련셋/시험셋')
    st.write('test 파일에 종속변수인 Price컬럼이 비어있어서(NaN) 데이터 전처리 불가능')
    st.image('https://github.com/skfkeh/MachineLearning/blob/main/img/train_test_error.PNG?raw=true', caption="test_file_NaN")
    st.write('해결책 : train 데이터 자체에서 훈련셋/시험셋을 구분')
    st.markdown('---')
    
    st.subheader('3. session error')
    st.write('button 혹은 tab 전환 등의 이벤트가 발생 시마다')
    st.write('streamlit 전체 소스가 재실행되어 balloon 함수가 반복 실행되었음')
#     st.image('')
        
    st.write('해결책 : streamlit에서 지원하는 session 함수가 존재했으며')
    st.write('   화면이 재실행되어도 이전 session의 값을 가져와 function이 재실행되는 것을 막을수 있었다')
    code_session = '''if 'chk_balloon' not in st.session_state:
    st.session_state['chk_balloon'] = False'''
    st.code(code_session, language='python')
    
    st.subheader('4. 각 알고리즘 합치기')
    st.write('각 알고리즘의 key 값이 충돌해서 plotly 그래프가 그려지지 않는 문제 발생')
    st.image('https://github.com/skfkeh/MachineLearning/blob/main/img/merge%20algorithm_key%20error.png?raw=true', caption="streamlit error 화면")
    st.write('해결책 : 각 알고리즘의 고유 변수명으로 변경 및 key값 부여')
    code_merge_error = '''keys = random.sample(range(1000, 9999), 3)'''
    st.code(code_merge_error, language='python')
    st.markdown('---')
    
    st.subheader('5. streamlit - xgboost')
    st.write('streamlit에서 xgboost 모델이 없다는 오류 발생')
    st.image('https://github.com/skfkeh/MachineLearning/blob/main/img/xgb_error.png?raw=true', caption="streamlit_xgboost_error")
    st.write('해결책 : ')

    
    
