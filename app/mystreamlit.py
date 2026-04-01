import streamlit as st
import pandas as pd
import joblib
import os

def basic_cleaning(x_input):
    x_out = x_input.copy()
    x_out['was_contacted'] = (x_out['pdays'] != -1).astype(int)
    x_out['pdays'] = x_out['pdays'].replace(-1, 0)
    return x_out

model = joblib.load(os.path.join(os.path.dirname(__file__), '../models/modelo_final.joblib'))

st.title('Bank Deposit Subscription Predictor')
st.markdown('Fill in the client information to predict if they will subscribe to a term deposit.')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Numeric')
    age = st.number_input('Age', min_value=18, max_value=100)
    balance = st.number_input('Balance', min_value=-10000, max_value=100000)
    day = st.number_input('Day of month', min_value=1, max_value=31)
    duration = st.number_input('Call duration (seconds)', min_value=1, max_value=5000)
    campaign = st.number_input('Number of contacts this campaign', min_value=1, max_value=50)
    previous = st.number_input('Previous contacts', min_value=0, max_value=50)
    pdays = st.number_input('Days since last contact (-1 if never)', min_value=-1, max_value=1000)

with col2:
    st.subheader('Categorical')
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                                'management', 'retired', 'self-employed', 'services',
                                'student', 'technician', 'unemployed', 'unknown'])
    marital = st.selectbox('Marital status', ['divorced', 'married', 'single'])
    education = st.selectbox('Education', ['primary', 'secondary', 'tertiary', 'unknown'])
    default = st.selectbox('Has credit in default?', ['no', 'yes'])
    housing = st.selectbox('Has housing loan?', ['no', 'yes'])
    loan = st.selectbox('Has personal loan?', ['no', 'yes'])
    contact = st.selectbox('Contact type', ['cellular', 'telephone', 'unknown'])
    month = st.selectbox('Last contact month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    poutcome = st.selectbox('Previous campaign outcome', ['failure', 'other', 'success', 'unknown'])

if st.button('Predict', use_container_width=True):
    input_data = pd.DataFrame([{
        'age': age, 'balance': balance, 'day': day, 'duration': duration,
        'campaign': campaign, 'previous': previous, 'pdays': pdays,
        'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan,
        'contact': contact, 'month': month, 'poutcome': poutcome
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    label = 'YES' if prediction == 1 else 'NO'

    st.divider()
    if prediction == 1:
        st.success(f'### Prediction: {label}')
    else:
        st.error(f'### Prediction: {label}')

    st.markdown(f'**Probability of subscribing: {probability:.2%}**')
    st.progress(probability)