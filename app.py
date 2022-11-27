# app.py
import streamlit as st
import pandas as pd
import time
from PIL import Image     # 이미지 처리 라이브러리

########### function ###########
def count_down(ts):
    with st.empty():
        input_time = 1*3
        while input_time>=0:
            minutes, seconds = input_time//60, input_time%60
            st.metric("Countdown", f"{minutes:02d}:{seconds:02d}")
            time.sleep(1)
            input_time -= 1
        st.empty()

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

file_name = 'titanic.csv'
url = f'https://raw.githubusercontent.com/skfkeh/regression/main/data/{file_name}'

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

jpg_url = "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/00f3d481-97e5-4de9-bcf2-48c82b265793/d7uteu8-e50dde9e-b8af-4fea-ab31-b7748470dc8b.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzAwZjNkNDgxLTk3ZTUtNGRlOS1iY2YyLTQ4YzgyYjI2NTc5M1wvZDd1dGV1OC1lNTBkZGU5ZS1iOGFmLTRmZWEtYWIzMS1iNzc0ODQ3MGRjOGIuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.X7DaOWcJkNe2H8jjTNtybdRCV9p5u4H_yFaOk7kMbFg"
st.image(jpg_url, caption="Why So Serious??!")

st.write(f"사용한 데이터 URL : {url}")








choice = st.selectbox("분석 알고리즘을 골라주세요", ["Logistic", "RandomForest", "XGBoost"])

if choice == 'Logistic':
    ts_number = st.slider(label="test_size를 설정해주세요",
                          min_value=0.00, max_value=1.00,
                          step=0.20, format="%f")
    st.write('Test_size : ', ts_number)
    
    btn_chkbox_rs = st.checkbox("random_state 설정")
    
    if btn_chkbox_rs:
        rs_number = st.slider("random_state 설정",
                              min_value=0, max_value=200,
                              step=50, format="%d")




    
if choice == 'Logistic' and SearchBtn:
    st.session_state['chk_strline'] = 'Logistic'

if choice == 'RandomForest' and SearchBtn:
    st.session_state['chk_strline'] = 'RandomForest'

if choice == 'XGBoost' and SearchBtn:
    st.session_state['chk_strline'] = 'XGBoost'


SearchBtn = st.button('Search')




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
