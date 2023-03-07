
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import re
import shap
import lime
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import streamlit as st

font_title = {"family": "serif", "color": "#0b5394", "weight": "bold","size": 10}
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
#with open('/project7_final_04032023/style.css') as f:
#    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
plt.style.use('fivethirtyeight')
plt.rcParams.update({'xtick.labelsize':14, 'ytick.labelsize':14,  'axes.labelsize': 14, 'legend.fontsize': 14,
                     'axes.titlesize': 18, 'axes.titleweight':'bold', 'axes.titleweight':'bold' }) 
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)

#============================================== Dashboards =============================================
st.title('Credit Scoring App')
st.markdown('this app is built using Streamlit, for the purpose of credit scoring')

@st.cache_data
def nan_check(data):
    '''Check Missing Values'''
    total = data.isnull().sum()
    percent_1 = data.isnull().sum() / data.isnull().count() * 100
    percent_2 = (np.round(percent_1, 2))
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%NAN']).sort_values('%NAN', ascending=False)
    return missing_data

@st.cache_data
def drop_nan (df, perc=10.0):
    min_count = int(((100 - perc) / 100) * df.shape[0] + 1)
    mod_df = df.dropna(axis=1, thresh=min_count)
    return mod_df

@st.cache_data
def nan_imputer(df):
    imputer = SimpleImputer(strategy='median')
    df.iloc[:,:] = imputer.fit_transform(df)
    df = pd.DataFrame(df)
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)
    df['TARGET'] = df['TARGET'].astype(int)
    return df

# load data
@st.cache_data
def get_data(path):
    df = pd.read_csv(path)
    return df
    
# ml_data_path = '/project7_final_04032023/data_train.csv'
ml_data_path = 'data_train.csv'
infos_data_path = 'app_data.csv'
models_path = 'models/'

data_train = get_data(ml_data_path)
infos_data = pd.read_csv(infos_data_path)

# Cette function prepare le dataset original, sans SMOTE, et on peut choisir de scaling ou pas.
@st.cache_data
def prepare_data(df=data_train, threshold=19.0, scaling=True):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_new = drop_nan(df, perc=threshold)
    data_new = nan_imputer(data_new)
    unscale_cols = ['TARGET', 'SK_ID_CURR']
    cols = list(data_new.columns)
    scale_cols = list(set(cols) - set(unscale_cols))
    data_new0 = data_new[data_new['TARGET'] == 0]
    data_new1 = data_new[data_new['TARGET'] == 1]
    data_new0_train, data_new0_test = train_test_split(data_new0, test_size=0.50, random_state=42)
    data_new1_train, data_new1_test = train_test_split(data_new1, test_size=0.10, random_state=42)
    data_new_train = pd.concat([data_new0_train, data_new1_train])
    data_new_test = pd.concat([data_new0_test, data_new1_test])
    data_new_train = data_new_train.sort_values('SK_ID_CURR')
    data_new_test = data_new_test.sort_values('SK_ID_CURR')
    if scaling:
        data_new_train[scale_cols] = scaler.fit_transform(data_new_train[scale_cols]) 
        data_new_test[scale_cols] = scaler.fit_transform(data_new_test[scale_cols])
    return data_new_train, data_new_test

data_new_train, data_new_test = prepare_data()
data = pd.concat([data_new_train, data_new_test])
X_train = data_new_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
X_test = data_new_test.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y_train = data_new_train[['TARGET']]
y_test = data_new_test[['TARGET']]

client_ids = infos_data['SK_ID_CURR'].unique()
# st.sidebar.selectbox(label='select or enter client id', options=client_ids)
st.sidebar.header("User Input")
client = st.sidebar.text_input("Enter Client ID: SK_ID_CURR 100002 to-", 100002)
#if client not in client_ids:
#    st.write("Client ID not found")
client_id = int(client)

st.markdown(f'you have selected client_id: {client_id}')
st.write(infos_data.head(2))
@st.cache_data
def get_client_id(client=100002):
    if client not in infos_data['SK_ID_CURR'].unique():
        st.error('Client ID not found')
    id_client = int(client)
    return id_client

client_id = get_client_id(client_id)

client_data = infos_data[infos_data['SK_ID_CURR']==client_id]
st.write(client_data.head(2))

st.markdown('### Some important client information')
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
def select_features(features, df_train=X_train, df_test=X_test):
    df_train_new = df_train[features]
    df_test_new = df_test[features]
    return df_train_new, df_test_new

# select model
modelname = st.sidebar.selectbox('select model', options=['LGBM', 'CATBOOST', 'RandomForest'])
c_weight = {0: 0.35, 1:0.65}

@st.cache_data
def get_model(model_name='LogisticRegression'):
    model_clf = LogisticRegression(class_weight = c_weight,C = 1., max_iter=200)
    if model_name=='LGBM':
        model_clf = LGBMClassifier(class_weight = c_weight, n_estimators=200, learning_rate=0.05)
    elif model_name=='CATBOOST':
        model_clf =CatBoostClassifier(iterations=300, learning_rate=0.05,logging_level='Silent')
    elif model_name=='RandomForest':
        model_clf = RandomForestClassifier(class_weight = c_weight, max_depth=50, n_estimators=200)
    return model_clf

model = get_model(modelname)

model.fit(X_train, y_train)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
feature_rank = pd.DataFrame({'Feature':X_train.columns, 'Value':model.feature_importances_}).sort_values(by="Value", ascending=False)
feature_rank = feature_rank[feature_rank['Value']>0.03] # modifiable
feature_rank = feature_rank.sort_values('Value', ascending =False)
features = feature_rank['Feature'].to_list()

x_train_select, x_test_select = select_features(features)
data_select = pd.concat([x_train_select, x_test_select])
idx = client_data.index[0]
cl = data_select[data_select.index==idx]

model.fit(x_train_select, y_train)
roc_auc = roc_auc_score(y_test, model.predict_proba(x_test_select)[:,1])
st.markdown(f'Model Score: {roc_auc:.4f}')

# shap.initjs()
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(x_train_select)
# shap.summary_plot(shap_values, x_train_select)
# st.pyplot()

# @st.cache_data
# def get_client_exp():
#     exp_client = data[data['SK_ID_CURR']==client_id]
#     exp_client = exp_client[features]
#     X = exp_client.iloc[0]
#     return X

explainer = lime.lime_tabular.LimeTabularExplainer(training_data=x_train_select.values,
                                                   feature_names=x_train_select.columns,
                                                   class_names=['Accepted', 'Refused'],
                                                   mode='classification')

exp = explainer.explain_instance(data_row=data_select.iloc[idx], predict_fn=model.predict_proba, num_features=15)
exp.as_pyplot_figure()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# exp_client = get_client_exp()
# exp = explainer.explain_instance(data_row=exp_client, predict_fn=model.predict_proba)
# exp.show_in_notebook(show_table=True)

# exp1, exp2 = st.beta_columns(2)
# shap.summary_plot(shap_values, x_train_select)
# plt.savefig('scratch.png')
prob = model.predict_proba(cl)
st.markdown('### Predictions')
p1, p2 = st.columns(2)
with p1:
    st.metric(label='Accepted', value=np.round(prob[0][0], 3))
with p2:
    st.metric(label='Refused', value=np.round(prob[0][1], 3))