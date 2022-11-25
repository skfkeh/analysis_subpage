# app.py
import streamlit as st
import pandas as pd
import time
from PIL import Image     # 이미지 처리 라이브러리

########### function ###########
def count_down(ts):
    with st.empty():
        input_time = 1*5
        while input_time>=0:
            minutes, seconds = input_time//60, input_time%60
            st.metric("Countdown", f"{minutes:02d}:{seconds:02d}")
            time.sleep(1)
            input_time -= 1
        st.empty()

        
def Logistic_algorithm(url):
    df_train = pd.read_csv(url, index_col=0)

    #### 결측치 처리
    ## 1. Embarked 처리
    df_train.Embarked.fillna('S', inplace=True)

    ## 2. Cabin, Ticket Drop 처리
    df_train.drop(columns=['Cabin', 'Ticket'], inplace=True)
    df_train['Title'] = df_train.Name.str.extract('([a-zA-Z]+)\.')

    ## 3. Age 처리
    # Name에서 호칭 정리
    title_unique = df_train.Title.unique()
    rarelist = []
    for t in title_unique:
      if list(df_train.Title).count(t) < 10:
        rarelist.append(t)

    # 나이 평균 정의
    title_age_mean = df_train.groupby(['Title'])['Age'].mean()
    for t in df_train.Title.unique():
      df_train.loc[(df_train.Age.isnull()) & (df_train.Title == t), 'Age'] = title_age_mean[t]

    # Name, Title 컬럼 Drop
    df_train.drop(columns=['Name', 'Title'], inplace=True)


    ## 4. Sex, Embarked - dummy 화
    df_train_dummy = pd.get_dummies(df_train, columns=['Sex', 'Embarked'], drop_first=True)
    
    return df_train_dummy

########### function ###########
        
    
########### session ###########

if 'chk_balloon' not in st.session_state:
    st.session_state['chk_balloon'] = False

if 'chk_strline' not in st.session_state:
    st.session_state['chk_strline'] = ''
    
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = ''
    
########### session ###########
       

########### define ###########

# file_name = 'titanic.csv'
# url = f'https://raw.githubusercontent.com/skfkeh/regression/main/data/{file_name}'

########### define ###########
    
if st.session_state['chk_balloon'] == False:
    count_down(5)
    with st.spinner(text="Please wait..."):
        time.sleep(1)

    st.balloons()
    st.session_state['chk_balloon'] = True


################################
#####                      #####
#####       UI Start       #####
#####                      #####
################################

st.title('내 항공료는 왜 비싼 것인가')

url = "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/00f3d481-97e5-4de9-bcf2-48c82b265793/d7uteu8-e50dde9e-b8af-4fea-ab31-b7748470dc8b.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzAwZjNkNDgxLTk3ZTUtNGRlOS1iY2YyLTQ4YzgyYjI2NTc5M1wvZDd1dGV1OC1lNTBkZGU5ZS1iOGFmLTRmZWEtYWIzMS1iNzc0ODQ3MGRjOGIuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.X7DaOWcJkNe2H8jjTNtybdRCV9p5u4H_yFaOk7kMbFg"
st.image(url, caption="Why So Serious??!", width="100%")

btn_choice = st.radio("분석 알고리즘을 골라주세요",
               ("Logistic", "RandomForest", "XGBoost"))
SearchBtn = st.button('Search')



    
if btn_choice == 'Logistic' and SearchBtn:
    st.session_state['chk_strline'] = 'Logistic'

if btn_choice == 'RandomForest' and SearchBtn:
    st.session_state['chk_strline'] = 'RandomForest'

if btn_choice == 'XGBoost' and SearchBtn:
    st.session_state['chk_strline'] = 'XGBoost'

st.write(st.session_state['chk_strline'])

btn_chkbox = st.checkbox("WebCam")
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    


if btn_chkbox:
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer as a 3D uint8 tensor with PyTorch:
        bytes_data = img_file_buffer.getvalue()
        torch_img = torch.ops.image.decode_image(
            torch.from_numpy(np.frombuffer(bytes_data, np.uint8)), 3
        )

        # Check the type of torch_img:
        # Should output: <class 'torch.Tensor'>
        st.write(type(torch_img))

        # Check the shape of torch_img:
        # Should output shape: torch.Size([channels, height, width])
        st.write(torch_img.shape)
