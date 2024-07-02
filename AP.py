import streamlit as st
import pandas as pd
import joblib

# 设置页面标题
# st.title('An Explainable Prediction Model for Adverse Outcomes in Severe Scrub Typhus')

# 使用 st.columns 创建两列布局
left_column, right_column = st.columns(2)

# 在左侧列放置侧边栏的输入选项
with left_column:
    # st.header('Variables')
    a = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0, max_value=1000, step=1)
    b = st.number_input("Total bilirubin (umol/L)", min_value=0.0, max_value=1000.0, step=0.1)
    c = st.number_input("Albumin (g/L)", min_value=0.0, max_value=100.0, step=0.1)
    d = st.selectbox("Central Nervous System Involvement", ("No", "Yes"))
    e = st.selectbox("Eschar", ("No", "Yes"))
    f = st.selectbox("Dyspnea", ("No", "Yes"))
    g = st.selectbox("Pulmonary Infection", ("No", "Yes"))
    h = st.number_input("Total protein (g/L)", min_value=0.0, max_value=1000.0, step=0.1)
    i = st.number_input("Total blood calcium (mmol/L)", min_value=0.0, max_value=1000.0, step=0.1)
    j = st.number_input("White blood cell count (10^9/L)", min_value=0.0, max_value=1000.0, step=0.1)

# 如果按下按钮
if right_column.button("Predict"):  # 在右侧列放置预测按钮
    # 加载训练好的模型
    model = joblib.load("XGBoost.pkl")

    # 将输入存储为 DataFrame，确保列名与模型训练时一致
    X = pd.DataFrame([[a, b, c, 1 if d == 'Yes' else 0, 1 if e == 'Yes' else 0,
                       1 if f == 'Yes' else 0, 1 if g == 'Yes' else 0, h, i, j]],
                     columns=['Systolic Blood Pressure(mmHg)', 'Total bilirubin(umol/L)',
                              'Albumin(g/L)', 'Central Nervous System Involvement', 'Eschar',
                              'Dyspnea', 'Pulmonary Infection', 'Total protein(g/L)',
                              'Total blood calcium(mmol/L)', 'White blood cell count(10^9/L)'])

    # 创建一个空白容器来放置预测结果
    result_container = st.empty()

    # 进行预测
    prediction = model.predict(X)[0]
    Predict_proba = model.predict_proba(X)[:, 1][0]

    # 输出预测结果到空白容器
    result_container.subheader(f"Probability of predicting adverse outcome: {'%.2f' % (Predict_proba * 100)}%")
