import pickle
import streamlit as st
import time

from pathlib import Path
import xgboost as xgb

#st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

footerText = """
<style>
#MainMenu {
visibility:hidden ;
}

footer {
visibility : hidden ;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: transparent;
color: white;
text-align: center;
}
</style>

<div class='footer'>
<p> Copyright @ 2023 Center for Digital Health <a href="mailto:iceanon1@khu.ac.kr"> iceanon1@khu.ac.kr </a></p>
</div>
"""

st.markdown(str(footerText), unsafe_allow_html=True)

@st.cache_data
#sub_finalized_model_adb predict_substance_model
def model_file():
    mfile = str(Path(__file__).parent) + '/retinopathy.pkl'
    with open(mfile, 'rb') as file:
        model = pickle.load(file)
    return model

# predict_substance_model
# sub_finalized_model_lgb


def prediction(X_test):
    model = model_file()
    result = model.predict_proba([X_test])

    return result[0][1]

def input_values():
    age = st.number_input('Age', min_value=10, max_value=100, value=30)
    
    sex = st.radio('Sex', ('Male', 'Female'), horizontal=True)
    sexDict = {'Male': 1, 'Female': 2}
    sex = sexDict[sex]

    # 연속형 변수 입력 받기
    HbA1c = st.number_input('HbA1c (median)', min_value=3.9, max_value=10.8, value=5.0)
    Glucose = st.number_input('Glucose (median)', min_value=42.0, max_value=359.0, value=100.0)
    TC = st.number_input('TC (median)', min_value=49.0, max_value=282.0, value=200.0)
    TG = st.number_input('TG (median)', min_value=27.0, max_value=486.0, value=150.0)
    HDL = st.number_input('HDL (median)', min_value=10.0, max_value=89.0, value=50.0)
    LDL = st.number_input('LDL (median)', min_value=14.5, max_value=187.0, value=100.0)
    Cr = st.number_input('Cr (median)', min_value=0.2, max_value=4.1, value=1.0)
    AST = st.number_input('AST (median)', min_value=8.0, max_value=237.5, value=30.0)
    ALT = st.number_input('ALT (median)', min_value=1.0, max_value=147.0, value=30.0)
    GGT = st.number_input('GGT (median)', min_value=5.0, max_value=309.0, value=30.0)
    ALP = st.number_input('ALP (median)', min_value=16.5, max_value=224.0, value=70.0)
    BMI = st.number_input('BMI (median)', min_value=12.0, max_value=39.3, value=25.0)

    # 표준편차 변수 입력 받기 (std)
    HbA1c_std = st.number_input('HbA1c (std)', min_value=0.0, max_value=3.7476659403, value=1.0)
    Glucose_std = st.number_input('Glucose (std)', min_value=0.0, max_value=207.18228689, value=50.0)
    TC_std = st.number_input('TC (std)', min_value=0.0, max_value=119.50104602, value=30.0)
    TG_std = st.number_input('TG (std)', min_value=0.0, max_value=331.63308038, value=80.0)
    HDL_std = st.number_input('HDL (std)', min_value=0.0, max_value=53.033008589, value=15.0)
    LDL_std = st.number_input('LDL (std)', min_value=0.0, max_value=94.752308679, value=25.0)
    Cr_std = st.number_input('Cr (std)', min_value=0.0, max_value=2.2627416998, value=0.5)
    AST_std = st.number_input('AST (std)', min_value=0.0, max_value=307.59144982, value=50.0)
    ALT_std = st.number_input('ALT (std)', min_value=0.0, max_value=113.13708499, value=30.0)
    GGT_std = st.number_input('GGT (std)', min_value=0.0, max_value=200.81832586, value=50.0)
    ALP_std = st.number_input('ALP (std)', min_value=0.0, max_value=115.96551211, value=30.0)
    BMI_std = st.number_input('BMI (std)', min_value=0.0, max_value=11.596551211, value=3.0)

    # 이진 변수 입력 받기
    binary_cols = ['met', 'sul', 'dpp', 'mg', 'thia', 'glu', 'insul', 'glp', 'sglt', 'angio', 'convert', 'cal', 'di', 
                   'beta', 'sta', 'fi', 'eze', 'aspi', 'clo', 'cil', 'gly', 'hypertension', 'dyslipidemia', 
                   'parkinson', 'demen', 'kidney', 'endstage', 'neuro', 'malignant', 'CCD', 'PL']
    binary_inputs = []
    for col in binary_cols:
        value = st.radio(col, ('0', '1'), horizontal=True)
        binary_inputs.append(int(value))

    # 모든 입력 값을 하나의 리스트로 결합
    X_test = [age, sex, HbA1c, Glucose, TC, TG, HDL, LDL, Cr, AST, ALT, GGT, ALP, BMI,
              HbA1c_std, Glucose_std, TC_std, TG_std, HDL_std, LDL_std, Cr_std,
              AST_std, ALT_std, GGT_std, ALP_std, BMI_std] + binary_inputs

    return X_test


def main():
    st.title("Possibility of Retinopathy outcome within 3 years")

    # 사용자 입력 받기
    X_test = input_values()

    # 예측 확률 계산
    prediction_result = prediction(X_test)

    # 실제 모델에 따라 이 부분은 수정이 필요할 수 있습니다.
    probability = prediction_result[0]  # 예를 들어, 결과가 이미 확률로 반환된다고 가정
    probability = round(probability * 100, 2)  # 백분율로 변환 및 반올림

    # 확률 결과 표시
    st.markdown(f'## 예측 확률')
    st.markdown(f'이 환자의 예측 확률은 **{probability}%** 입니다.')

    # 사이드바에 이미지 표시
    with st.sidebar:
        img2 = Image.open('img_1.png')
        st.image(img2)

        # 사이드바에 확률 결과 표시
        st.markdown(f'### 예측된 확률')
        st.markdown(f'#### {probability}%')

    # 현재 시간 출력 (디버깅 용도)
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    print(now)
       

if __name__ == '__main__':
    main()
