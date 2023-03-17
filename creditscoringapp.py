import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lime
from lime import lime_tabular
import pickle
import streamlit as st
#============================================== Dashboards =============================================
st.title('Credit Scoring App')
st.markdown('#### this app is built using Streamlit, for the purpose of credit scoring')

data_path = 'data/test_data.csv'
infos_data_path = 'data/clients_infos.csv'
models_path = 'models/'
#---
# load data
@st.cache_data
def get_data(path):
    df = pd.read_csv(path)
    return df

clients_info = get_data(infos_data_path)
test_data = get_data(data_path)

client_ids = test_data['SK_ID_CURR'].unique()

st.sidebar.header("User Input")

client = st.sidebar.selectbox(label='select or enter client id', options=client_ids)
client_id = int(client)
st.write(test_data.head(2))

st.markdown(f'you have selected client_id: {client_id}')

client_data = clients_info[clients_info['SK_ID_CURR']==client_id]
st.write(client_data.head())
idx = client_data.index[0]
cl = test_data[test_data.index==idx]

st.markdown('#### Some important client information')
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label='Age', value=client_data['AGE']) 
with col2:
    st.metric(label='Gender', value=client_data['CODE_GENDER'].unique()[0])   
with col3:
    st.metric(label='Status', value=client_data['NAME_FAMILY_STATUS'].unique()[0])
with col4:
    st.metric(label='Income', value=client_data['AMT_INCOME_TOTAL'])
with col5:
    st.metric(label='Occupation', value=client_data['OCCUPATION_TYPE'].unique()[0])
    
@st.cache_data
def select_features(features, df=test_data):
    df_select = df[features]
    return df_select

# select model
modelname = st.sidebar.selectbox('select model', options=['LGBM'])

@st.cache_data
def load_model():
    model = pickle.load(open('models/lgbm.pickle' , 'rb'))
    return model

model = load_model()

x_test_select = test_data.drop('SK_ID_CURR', axis=1)
cl = cl.drop('SK_ID_CURR', axis=1)

explainer = lime_tabular.LimeTabularExplainer(training_data=x_test_select.values,
                                                   feature_names=x_test_select.columns,
                                                   class_names=['Accepted', 'Refused'],
                                                   mode='classification')

exp = explainer.explain_instance(data_row=x_test_select.iloc[idx], predict_fn=model.predict_proba, num_features=15)
exp.as_pyplot_figure()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

prob = model.predict_proba(cl)
st.markdown('### Predictions')
p1, p2 = st.columns(2)
with p1:
    st.metric(label='Accepted', value=np.round(prob[0][0], 4))
with p2:
    st.metric(label='Refused', value=np.round(prob[0][1], 4))
    
    
#=============================================================End of Dashboards ============================================================