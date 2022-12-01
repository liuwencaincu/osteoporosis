import streamlit as st
from pycaret.classification import *
import pandas as pd
import numpy as np


#应用主题
st.set_page_config(
    page_title="ML Medicine",
    #page_icon="🐇",
)
#隐藏选项卡
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


#应用标题
st.title('Machine Learning Application for Predicting Bone Metastasis in TNBC')

male_gender = st.sidebar.selectbox("male_gender",('No','Yes'))
age = st.sidebar.slider("age",0,100,50,1)
Hypertension = st.sidebar.selectbox("Hypertension",('No','Yes'))
CHD = st.sidebar.selectbox("CHD",('No','Yes'))
Lipid_disorder = st.sidebar.selectbox("Lipid_disorder",('No','Yes'))
Stroke = st.sidebar.selectbox("Stroke",('No','Yes'))
Heart_failure = st.sidebar.selectbox("Heart_failure",('No','Yes'))
Cancer = st.sidebar.selectbox("Cancer",('No','Yes'))
Diabetes = st.sidebar.selectbox("Diabetes",('No','Yes'))
COPD = st.sidebar.selectbox("COPD",('No','Yes'))
Chronic_kidney_disease = st.sidebar.selectbox("Chronic_kidney_disease",('No','Yes'))

#映射字典
map = {'No':0,'Yes':1}

#映射
male_gender = map[male_gender]
Hypertension = map[Hypertension]
CHD = map[CHD]
Lipid_disorder = map[Lipid_disorder]
Stroke = map[Stroke]
Heart_failure = map[Heart_failure]
Cancer = map[Cancer]
Diabetes = map[Diabetes]
COPD = map[COPD]
Chronic_kidney_disease = map[Chronic_kidney_disease]

#读之前存储的模型
model = load_model('saved_stacker')

input_dict = {'male_gender':male_gender, 'age':age, 'Hypertension':Hypertension,
              'CHD':CHD, 'Lipid_disorder':Lipid_disorder, 'Stroke':Stroke,
              'Heart_failure':Heart_failure, 'Cancer':Cancer,
              'Diabetes':Diabetes, 'COPD':COPD,
              'Chronic_kidney_disease':Chronic_kidney_disease,
              }
input_df = pd.DataFrame([input_dict])

#截断点
sp = 0.5
#figure
is_t = (model.predict_proba(input_df)[0][1])> sp
prob = (model.predict_proba(input_df)[0][1])*1000//1/10

#预测
if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability:  '+str(prob)+'%')
